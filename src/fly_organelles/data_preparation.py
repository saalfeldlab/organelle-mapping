import copy
import itertools
import logging
import os
from pathlib import Path
from typing import BinaryIO, Optional

import click
import fibsem_tools as fst
import numcodecs
import numpy as np
import skimage
import xarray as xr
import yaml
import zarr
from cellmap_schemas.annotation import (
    AnnotationArrayAttrs,
    AnnotationGroupAttrs,
    SemanticSegmentation,
    wrap_attributes,
)
from pydantic_zarr.v2 import ArraySpec, GroupSpec
from xarray_multiscale import multiscale, windowed_mode

#from fibsem_tools.io.multiscale import multiscale_group
from xarray_ome_ngff import create_multiscale_group

from fly_organelles.utils import (
    GeneralizedAnnotationArrayAttrs,
    SmoothSemanticSegmentation,
    all_combinations,
    find_target_scale,
    read_label_yaml,
    valid_offset,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.group()
def cli():
    pass


def filter_crops_for_sampling(datasets, sampling, labels):
    new_datasets = copy.deepcopy(datasets)
    for dataset, ds_info in datasets["datasets"].items():
        new_crops = []
        for crop in ds_info["labels"]["crops"]:
            myc = []
            for c in crop.split(","):
                try:
                    for label in labels:
                        find_target_scale(
                            fst.read(Path(datasets["labels"]["data"]) /datasets["labels"]["group"] / c / label), sampling
                        )
                except ValueError:
                    continue
                myc.append(c)
            if len(myc) > 0:
                new_crops.append(",".join(myc))
        if len(new_crops) > 0:
            new_datasets["datasets"][dataset]["labels"]["crops"] = new_crops
        else:
            del new_datasets["datasets"][dataset]
    return new_datasets


def filter_crops_for_percent_annotated(datasets, sampling, labels, threshold_percent_annotated=25):
    new_datasets = copy.deepcopy(datasets)
    for dataset, ds_info in datasets["datasets"].items():
        new_crops = []
        for crop in ds_info["labels"]["crops"]:
            myc = []
            for c in crop.split(","):
                for label in labels:
                    scale, _, shape = find_target_scale(
                        fst.read(Path(datasets["labels"]["data"]) / datasets["labels"]["group"] / c / label), sampling
                    )
                    ann_attrs = fst.read(
                        Path(datasets["labels"]["data"]) / datasets["labels"]["group"]/ c / label / scale
                    ).attrs["cellmap"]["annotation"]
                    voxels = np.prod(shape)
                    if "unknown" in ann_attrs["complement_counts"]:
                        not_annotated = ann_attrs["complement_counts"]["unknown"]
                    else:
                        not_annotated = 0
                    annotated = voxels - not_annotated
                    if annotated / voxels > threshold_percent_annotated / 100.0:
                        myc.append(c)
                        break
            if len(myc) > 0:
                new_crops.append(",".join(myc))
        if len(new_crops) > 0:
            new_datasets["datasets"][dataset]["labels"]["crops"] = new_crops
        else:
            del new_datasets["datasets"][dataset]
    return new_datasets


@cli.command()
@click.argument("data-config-in", type=click.File("rb"))
@click.argument("data-config-out", type=click.File("w"))
@click.option("--sampling", type=float, nargs=3, required=True)
@click.option("--label", type=click.STRING, multiple=True, required=True)
@click.option("--threshold_percent_annotated", type=float, default=25, show_default=True)
@click.option("--skip_filter_sampling", is_flag=True, default=False)
@click.option("--skip_filter_percent_annotated", is_flag=True, default=False)
def filter_crop_list(
    data_config_in,
    data_config_out,
    sampling,
    label,
    *,
    threshold_percent_annotated=25,
    skip_filter_sampling=False,
    skip_filter_percent_annotated=False,
):
    filter_crop_list_func(
        data_config_in,
        data_config_out,
        sampling,
        label,
        threshold_percent_annotated=threshold_percent_annotated,
        skip_filter_sampling=skip_filter_sampling,
        skip_filter_percent_annotated=skip_filter_percent_annotated,
    )


def filter_crop_list_func(
    data_config_in,
    data_config_out,
    sampling,
    label,
    *,
    threshold_percent_annotated=25,
    skip_filter_sampling=False,
    skip_filter_percent_annotated=False,
):
    data_dict = yaml.safe_load(data_config_in)
    if not skip_filter_sampling:
        data_dict = filter_crops_for_sampling(data_dict, sampling, label)
    if not skip_filter_percent_annotated:
        data_dict = filter_crops_for_percent_annotated(
            data_dict, sampling, label, threshold_percent_annotated=threshold_percent_annotated
        )
    yaml.safe_dump(data_dict, data_config_out, default_flow_style=False)


def verify_classes(classes: dict[str, set[str]]) -> tuple[bool, int]:
    atoms_col: list[str] = []
    for k, v in classes.items():
        if len(v) < 1:
            return False, 1  # empty label
        elif len(v) == 1:
            if k != next(iter(v)):
                return False, 2  # atomic label with label not matching atom
            atoms_col.append(k)
    if len(atoms_col) != len(set(atoms_col)):
        return False, 3  # atom defined twice
    atoms = set(atoms_col)
    for v in classes.values():
        if any(vv not in atoms for vv in v):
            return False, 4  # label composed of non-atomic labels
    hashable_values = [tuple(v) for v in classes.values()]
    if len(hashable_values) != len(set(hashable_values)):
        return False, 5  # several labels with same atoms
    return True, 0


class Crop:
    def __init__(self, classes: dict[str, set[str]], crop_path):
        if not verify_classes(classes)[0]:
            msg = "Classes dictionary is faulty"
            raise ValueError(msg)
        self.classes: dict[str, set[str]] = classes
        self.crop_path = crop_path
        self.crop_root = fst.access(self.crop_path, mode="r+")

    def get_shape(self, scale_level="s0") -> tuple[int]:
        some_label = next(iter(self.get_annotated_classes()))
        return self.crop_root[some_label][scale_level].shape

    def get_chunking(self, scale_level="s0") -> tuple[int]:
        some_label = next(iter(self.get_annotated_classes()))
        return self.crop_root[some_label][scale_level].chunks

    def get_coords(self, scale_level="s0"):
        some_label = next(iter(self.get_annotated_classes()))
        some_xarr = fst.read_xarray(os.path.join(self.crop_path, some_label, scale_level))
        return some_xarr.coords

    def get_annotated_classes(self) -> set[str]:
        return set(self.crop_root.attrs["cellmap"]["annotation"]["class_names"])
    def get_annotation_type(self,  label: str, scale_level="s0") -> str:
        return self.crop_root[label][scale_level].attrs["cellmap"]["annotation"]["annotation_type"]["type"]
    def get_attributes(self, label: Optional[str] = None, scale_level: Optional[str] = "s0") -> dict():
        attr_src = self.crop_root
        if label is not None:
            attr_src = attr_src[label]
            if scale_level is not None:
                attr_src = attr_src[scale_level]
        return attr_src.attrs.dict()
    def get_array(self, label: str, scale_level="s0") -> np.ndarray:
        return np.array(self.crop_root[label][scale_level])
    def get_encoding(self, label: str) -> dict[str, int]:
        return self.crop_root[label].attrs["cellmap"]["annotation"]["annotation_type"]["encoding"]
    def get_scalelevels(self, label: str) -> list[str]:
        return sorted(self.crop_root[label].keys(),key=lambda x: int(x[1:]))

    def create_new_class(self, atoms: set[str]):
        encoding = {"absent": 0, "unknown": 255, "present": 1}
        n_arr = encoding["unknown"] * np.ones(self.get_shape(), dtype=np.uint8)
        subcombos = []
        for l in self.get_annotated_classes():
            label_encoding = self.get_encoding(l)
            l_set = self.classes[l]
            if l_set == atoms:
                msg = "combination is already an annotated class"
                raise ValueError(msg)
            if l_set < atoms:
                n_arr[self.get_array(l) == label_encoding["present"]] = encoding["present"]
                if len(l_set) > 1:
                    subcombos.append(l)
            elif atoms.isdisjoint(l_set):
                n_arr[self.get_array(l) == label_encoding["present"]] = encoding["absent"]
        if atoms <= self.get_annotated_classes():
            n_arr[np.logical_and.reduce([self.get_array(a) == self.get_encoding(a)["absent"] for a in atoms])] = encoding["absent"]
        for combo in all_combinations(subcombos):
            if not np.any(n_arr == encoding["unknown"]):
                break
            missing = atoms - set().union(*(self.classes[c] for c in combo))
            if missing <= self.get_annotated_classes():
                n_arr[np.logical_and.reduce([self.get_array(c) == self.get_encoding(c)["absent"] for c in missing.union(combo)])] = encoding["absent"]
        return n_arr.astype(np.uint8), encoding

    def save_class(self, name: str, arr: np.ndarray, encoding: dict[str,int], overwrite: bool = False):
        xarr = xr.DataArray(arr, coords=self.get_coords())
        multi = {m.name: m for m in multiscale(xarr, windowed_mode, (2, 2, 2), chunks=self.get_chunking())}
        label_array_specs = {}
        # intialize some stuff for reuse
        annotation_type = SemanticSegmentation(encoding=encoding)
        compressor = numcodecs.Zstd(level=3)
        for mslvl, msarr in multi.items():
            # get complement counts for annotation metadata
            ids, counts = np.unique(msarr, return_counts=True)
            histo = {}
            if encoding["unknown"] in ids:
                histo["unknown"] = counts[list(ids).index(encoding["unknown"])]
            if encoding["absent"] in ids:
                histo["absent"] = counts[list(ids).index(encoding["absent"])]
            if encoding["present"] in ids:
                histo["present"] = counts[list(ids).index(encoding["present"])]
            # initialize array wise metadata
            annotation_array_attrs = AnnotationArrayAttrs(
                class_name=name, complement_counts=histo, annotation_type=annotation_type
            )
            label_array_specs[mslvl] = wrap_attributes(annotation_array_attrs).dict()
            # label_array_specs[mslvl] = ArraySpec.from_array(
            #     msarr,
            #     chunks=self.get_chunking(),
            #     compressor=compressor,
            #     attributes=wrap_attributes(annotation_array_attrs).dict(),
            # )
        # intialize group attributes for annotation
        annotation_group_attrs = AnnotationGroupAttrs(class_name=name, description="", annotation_type=annotation_type)
        # intialize group attributes for multiscale
        ms_group = create_multiscale_group(
             store = self.crop_root.store,
             path = os.path.join(self.crop_root.path, name),
             arrays = multi,
             chunks=self.get_chunking(),
             compressor=compressor,
             overwrite=overwrite
         )
        ms_group.attrs.update(wrap_attributes(annotation_group_attrs).dict())
        for mslvl, msarr in ms_group.items():
            msarr.attrs.update(label_array_specs[mslvl])


        for mslvl, msarr in multi.items():
            ms_group[mslvl][:] = msarr.data

        if name not in self.get_annotated_classes():
            atts = self.crop_root.attrs
            atts["cellmap"]["annotation"]["class_names"] = atts["cellmap"]["annotation"]["class_names"] + [name]
            self.crop_root.attrs.update(atts)

    def add_new_class(self, name: str, atoms: Optional[set[str]] = None):
        if atoms is None:
            atoms = self.classes[name]
        elif name in self.classes and self.classes[name] != atoms:
            msg = "A class with name {name} exists in the configured set of classes, but the atoms do not match."
            raise ValueError(msg)
        arr, encoding = self.create_new_class(atoms)
        self.save_class(name, arr, encoding)

    def convert_to_semantic(self, label: Optional[str]=None):
        if label is None:
            for lbl in self.get_annotated_classes():
                self.convert_to_semantic(lbl)
        elif self.get_annotation_type(label) == "semantic_segmentation":
            logger.info(f"Label {label} is already semantic segmentation")
            return
        else:
            logger.info(f"Converting {label} to semantic segmentation")
            arr = self.get_array(label).copy()
            label_encoding = self.get_encoding(label)
            present_mask = np.logical_not(np.logical_or(arr == label_encoding["absent"], arr == label_encoding["unknown"]))
            encoding = {"absent": 0, "unknown": 255, "present": 1}
            arr[present_mask] = encoding["present"]
            if encoding["unknown"] != label_encoding["unknown"]:
                arr[arr == label_encoding["unknown"]] = encoding["unknown"]
            if encoding["absent"] != label_encoding["absent"]:
                arr[arr == label_encoding["absent"]] = encoding["absent"]
            self.save_class(label, arr, encoding, overwrite=True)
    def smooth_multiscale(self, label: Optional[str]=None):
        if label is None:
            for lbl in self.get_annotated_classes():
                self.smooth_multiscale(lbl)
        else:
            logger.info(f"Generating smooth multiscale for {label}")
            if self.get_annotation_type(label) != "semantic_segmentation":
                msg = f"Label {label} is not semantic segmentation, convert to semantic segmentation first"
                raise ValueError(msg)
            encoding = self.get_encoding(label)
            if encoding["unknown"] <= 8 * max(encoding["present"], encoding["absent"]):
                msg = f"smoothing relies on large value for encoding unknown, found values {encoding} for {label} in {cropname}"
                raise ValueError(msg)
            full_scale = self.get_scalelevels()[0]
            full_scale_arr = self.get_array(label, full_scale)
            multi_valued = np.unique(full_scale_arr).size != 1
            for l1, l2 in itertools.pairwise(self.get_scalelevels()):
                src = self.get_array(label, l1)
                down = skimage.transform.downscale_local_mean(src, 2).astype("float32")
                downslice = tuple(slice(sh) for sh in (np.array(down.shape) // 2) * 2)
                down = down[downslice]
                down[down > max(encoding["present"], encoding["absent"])] = encoding["unknown"]
                self.get_attributes(label, l2)
                histo = {}
                histo["absent"] = round(
                    np.sum(encoding["present"] - down[down != encoding["unknown"]]), 2)
                histo["unknown"] = np.sum(down == encoding["unknown"])
                #histo["present"] = np.prod(down.shape) - histo["absent"] - histo["unknown"]
                if multi_valued:
                    annotation_type = SmoothSemanticSegmentation(
                        encoding=encoding)
                    annotation_array_attrs = GeneralizedAnnotationArrayAttrs(
                        class_name=label,
                        complement_counts=histo,
                        annotation_type=annotation_type)
                else:
                    annotation_type = SemanticSegmentation(
                        encoding=encoding
                    )
                    annotation_array_attrs = AnnotationArrayAttrs(
                        class_name=label,
                        complement_counts=histo,
                        annotation_type=annotation_type)
                    down = down.astype(src.dtype)
                self.crop_root[label].create_dataset(
                    l2, data=down, overwrite=True, dimension_separator="/", compressor=numcodecs.Zstd(level=3)
                )
                self.crop_root[label][l2].attrs.update(
                    wrap_attributes(
                        annotation_array_attrs).model_dump())


@cli.command()
@click.argument("label-config", type=click.File("rb"))
@click.argument("data-config", type=click.File("rb"))
def smooth_multiscale(label_config: BinaryIO, data_config: BinaryIO):
    _smooth_multiscale(label_config, data_config)

def _smooth_multiscale(label_config: BinaryIO, data_config: BinaryIO):
    datas = yaml.safe_load(data_config)
    classes = read_label_yaml(label_config)
    for _, ds_info in datas["datasets"].items():
        for crops in ds_info["labels"]["crops"]:
            for cropname in crops.split(","):
                crop = Crop(classes, f"{ds_info['labels']['data']}/{ds_info['labels']['group']}/{cropname}")
                crop.smooth_multiscale()
@cli.command()
@click.argument("label-config", type=click.File("rb"))
@click.argument("data-config", type=click.File("rb"))
@click.argument("new-label", type=click.STRING)
def add_class_to_all_crops(label_config: BinaryIO, data_config: BinaryIO, new_label: str):
    """Script to add a new composite label to all crops in the given data configuration.

    Args:
        label_config (BinaryIO): Yaml file with label definitions. Should include the `new_label`
        data_config (BinaryIO): Yaml file describing groundtruth data.
        new_label (str): Name of new label to be added (as specified in `label_config`)
    """
    _add_class_to_all_crops_func(label_config, data_config, new_label)


def _add_class_to_all_crops_func(label_config: BinaryIO, data_config: BinaryIO, new_label: str):
    datas = yaml.safe_load(data_config)
    classes = read_label_yaml(label_config)
    for key, ds_info in datas["datasets"].items():
        logger.info(f"Processing {key}")
        for crop in ds_info["labels"]["crops"]:
            c = Crop(classes, f"{ds_info['labels']['data']}/{ds_info['labels']['group']}/{crop}")
            if new_label in c.get_annotated_classes():
                logger.info(f"Label {new_label} already exists in {crop}")
            else:
                c.add_new_class(new_label)

@cli.command()
@click.argument("label-config", type=click.File("rb"))
@click.argument("data-config", type=click.File("rb"))
@click.argument("label", type=click.STRING, required=False, default=None)
def convert_class_to_semantic(label_config: BinaryIO, data_config: BinaryIO, label: Optional[str] = None):
    """Script to convert labels from instance to semantic segmentation.

    Args:
        label_config (BinaryIO): Yaml file with label definitions.
        data_config (BinaryIO): Yaml file describing groundtruth data.
        label (Optional[str], optional): Name of label to convert. Defaults to None, 
            in which case all labels are converted.
    """
    _convert_class_to_semantic_func(label_config, data_config, label)
def _convert_class_to_semantic_func(label_config: BinaryIO, data_config: BinaryIO, label: Optional[str] = None):
    datas = yaml.safe_load(data_config)
    classes = read_label_yaml(label_config)
    for key, ds_info in datas["datasets"].items():
        logger.info(f"Processing {key}")
        for crop in ds_info["labels"]["crops"]:
            logger.info(f"Processing {crop}")
            c = Crop(classes, f"{ds_info['labels']['data']}/{ds_info['labels']['group']}/{crop}")
            c.convert_to_semantic(label)
@cli.command()
@click.argument("crop_path", type=str)
@click.option("--orig_offset", type=float, nargs=3)
@click.option("--new_offset", type=float, nargs=3)
@click.option("-r", "--rounding", type=int, default=-1)
def edit_offset(crop_path, orig_offset, new_offset, *, rounding: int = -1):
    edit_offset_func(crop_path, orig_offset, new_offset, rounding=rounding)


def edit_offset_func(crop_path, orig_offset, new_offset, *, rounding: int = -1):
    diff_offset = np.array(orig_offset) - np.array(new_offset)
    crop_root = fst.access(crop_path, mode="r+")
    for lbl in crop_root.keys():
        crop_label_root = crop_root[lbl]
        ms_attrs = crop_label_root.attrs["multiscales"]
        for ds in range(len(ms_attrs[0]["datasets"])):
            for k in range(len(ms_attrs[0]["datasets"][ds]["coordinateTransformations"])):
                if ms_attrs[0]["datasets"][ds]["coordinateTransformations"][k]["type"] == "translation":
                    transl = ms_attrs[0]["datasets"][ds]["coordinateTransformations"][k]["translation"]
                    set_offset = np.array(transl) - diff_offset
                    if rounding >= 0:
                        set_offset = np.round(set_offset, rounding)
                    ms_attrs[0]["datasets"][ds]["coordinateTransformations"][k]["translation"] = list(set_offset)
        crop_label_root.attrs["multiscales"] = ms_attrs


@cli.command()
@click.argument("data-config", type=click.File("rb"))
@click.option(
    "-n",
    "--dry-run",
    is_flag=True,
    help="Adding this flag will find all the offsets that can be fixed but doesn't edit any of them.",
)
def fix_offset(data_config: BinaryIO, *, dry_run: bool = False):
    """
    Script to check whether offsets of groundtruth crops are aligned with the grid of the raw data. If they are not,
    the script uses heuristics to guess what the correct offset might be and edits the crop metadata accordingly,
    unless the `dry_run` flag is added.

    Args:
        data_config (BinaryIO): Yaml file describing groundtruth data.
        dry_run (bool, optional): If True, the crop metadata will not be modified. Defaults to False.

    Returns:
        dict[str, list[str]]: summary dictionary with lists of crops that are already "valid",
            "correctable"/"corrected" and "invalid" (i.e. not automatically correctable).
    """
    datas = yaml.safe_load(data_config)
    summary: dict[str, list[str]] = {"valid": [], "invalid": []}
    if dry_run:
        summary["correctable"] = []
    else:
        summary["corrected"] = []
    for key, ds_info in datas["datasets"].items():
        logger.info(f"Processing {key}")
        root_raw = fst.read(Path(ds_info["em"]["data"]) / ds_info["em"]["group"])
        try:
            datasets = root_raw.attrs["multiscales"][0]["datasets"]
        except KeyError as e:
            logger.info("May have different metadata format. Proceeding with next dataset")
            continue
        for ds in datasets:
            if ds["path"] == "s0":
                raw_res = ds["coordinateTransformations"][0]["scale"]
        raw_res_arr = np.array(raw_res)
        for crop in ds_info["labels"]["crops"]:
            logger.info(f"Processing {ds_info['labels']['data']}/{ds_info['labels']['group']}/{crop}")
            crop_path =  Path(ds_info['labels']['data']) /ds_info['labels']['group'] / crop
            crop_root = fst.read(crop_path)
            lbl = next(iter(crop_root.keys()))
            crop_datasets = crop_root[lbl].attrs["multiscales"][0]["datasets"]
            for cds in crop_datasets:
                if cds["path"] == "s0":
                    crop_res = cds["coordinateTransformations"][0]["scale"]
                    crop_res_arr = np.array(crop_res)
                    crop_off = cds["coordinateTransformations"][1]["translation"]
                    crop_off_arr = np.array(crop_off)

                    if valid_offset(crop_off_arr, raw_res_arr, crop_res_arr):
                        logger.info(
                            f"Crop {crop} passed with raw resolution: "
                            f"{raw_res}, crop resolution {crop_res} and crop offset {crop_off}"
                        )
                        summary["valid"].append(crop)
                    else:
                        logger.info(
                            f"Crop {crop} did not pass with raw resolution: "
                            f"{raw_res}, crop resolution {crop_res} and crop offset {crop_off}"
                        )
                        if np.all(crop_res < raw_res):
                            crop_off_arr_suggested = crop_off_arr - crop_res_arr / 2.0

                        else:
                            crop_off_arr_suggested = crop_off_arr + crop_res_arr / 2.0 - raw_res_arr / 2.0
                        if valid_offset(crop_off_arr_suggested, raw_res_arr, crop_res_arr):
                            logger.info(f" Crop {crop} can be automatically corrected with {crop_off_arr_suggested}.")
                            if not dry_run:
                                edit_offset_func(crop_path, crop_off, list(crop_off_arr_suggested))
                                summary["corrected"].append(crop)
                            else:
                                summary["correctable"].append(crop)
                        else:
                            summary["invalid"].append(crop)
                            logger.info(f"Crop {crop} cannot be automatically corrected.")
    click.echo(summary)
    return summary
