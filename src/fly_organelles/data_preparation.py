import zarr
import numpy as np
import logging
import numcodecs
import yaml
import click
import fibsem_tools as fst
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@click.group()
def cli():
    pass

def create_label(crop_full_path: str, new_label_name: str, atomic_labels: list[str]):
    logger.info(f"Exporting {new_label_name} in {crop_full_path} by combining {atomic_labels}")
    crop = zarr.open(crop_full_path)
    levels: set | None = None
    shape: tuple | None = None
    multiscales: list | None = None
    if new_label_name in crop:
        assert crop[new_label_name]["s0"].attrs["cellmap"]["annotation"]["complement_counts"]["absent"] == np.prod(crop[new_label_name]["s0"].shape), f"{new_label_name} exists, not empty"
        logger.warning(f"Will overwrite {crop_full_path}/{new_label_name}")
    class_list = crop.attrs["cellmap"]["annotation"]["class_names"]    
    for atomic_label in atomic_labels:
        assert atomic_label in class_list, f"{atomic_label} not in list of classes ({class_list})"
        assert atomic_label in crop, f"Did not find {atomic_label} in {crop_full_path}"
        if levels is None:
            levels = set(crop[atomic_label].keys())
        else:
            levels = levels.intersection(crop[atomic_label].keys())
        if multiscales is None:
            multiscales = crop[atomic_label].attrs["multiscales"]
        else:
            assert multiscales == crop[atomic_label].attrs["multiscales"], f"Multiscales attribute of {atomic_label} does not match others"
    new_group = crop.create_group(new_label_name, overwrite=True)
    new_group.attrs["multiscales"] = multiscales
    #assert set(new_group.keys()).issubset(levels), f"There are more levels in {new_group} than in sources from {atomic_labels}"
    for level in sorted(levels):
        logger.info(f"Exporting {new_label_name}/{level}")
        for atomic_label in atomic_labels:
            if shape is None:
                shape = crop[atomic_label][level].shape
            else:
                assert (
                    shape == crop[atomic_label][level].shape
                ), f"Shape of {atomic_label}/{level} ({crop[atomic_label][level].shape})does not match other label shapes ({shape})"
        new_arr = np.zeros(shape, dtype=np.bool_)
        new_mask = np.ones(shape, dtype=np.bool_)
        cellmap_attrs = {

                "annotation": {
                    "annotation_type": {"encoding": {"absent": 0, "present": 1}, "type": "semantic_segmentation"},
                    "class_name": new_label_name,
                    "complement_counts": {},
                }
            }

        present_sum = 0
        for atomic_label in atomic_labels:
            present_id = crop[atomic_label][level].attrs["cellmap"]["annotation"]["annotation_type"]["encoding"][
                "present"
            ]
            if "unkown" in crop[atomic_label][level].attrs["cellmap"]["annotation"]["annotation_type"]["encoding"]:
                unknown_id = crop[atomic_label][level].attrs["cellmap"]["annotation"]["annotation_type"]["encoding"][
                    "unknown"
                ]
                bin_mask = np.array(crop[atomic_label][level]) == unknown_id
                np.logical_and(new_mask, bin_mask, out=new_mask)
            else:
                np.logical_and(new_mask, np.zeros(shape, dtype=np.bool_), out=new_mask)
            bin_atomic_label = np.array(crop[atomic_label][level]) == present_id
            logger.debug(f"Adding {atomic_label} with {np.sum(bin_atomic_label)}")
            np.logical_or(new_arr, bin_atomic_label, out=new_arr)
            logger.debug(f"After adding sum is {np.sum(new_arr)}")
            present_sum += np.prod(shape) - sum(
                v for k, v in crop[atomic_label][level].attrs["cellmap"]["annotation"]["complement_counts"].items()
            )
        assert present_sum == np.sum(
            new_arr
        ), f"Sum of present values from attributes ({present_sum})and in data ({np.sum(new_arr)}) do not match. Overlapping atomic labels?"
        new_arr = new_arr.astype(np.uint8)
        if np.sum(new_mask) != 0:
            logger.info(f"Mask is non-zero, setting to 255.")
            new_arr[new_mask] = 255
            cellmap_attrs["annotation"]["annotation_type"]["encoding"]["unknown"] = 255
            cellmap_attrs["annotation"]["complement_counts"]["unknown"] = np.sum(new_mask)
        cellmap_attrs["annotation"]["complement_counts"]["absent"] = np.sum(new_arr == 0)
        new_group.create_dataset(level, data=new_arr, fill_value=0, dimension_separator="/", compressor = numcodecs.Blosc(cname="zstd", clevel=5,shuffle=1), chunks=(64,64,64), overwrite=True)
        new_group[level].attrs["cellmap"] = cellmap_attrs
        shape = None
    del cellmap_attrs["annotation"]["complement_counts"]
    new_group.attrs["cellmap"] = cellmap_attrs
    class_list = crop.attrs["cellmap"]["annotation"]["class_names"]
    group_attrs = crop.attrs["cellmap"]
    if new_label_name not in class_list:
        class_list.append(new_label_name)
        group_attrs["annotation"]["class_names"] = class_list
        crop.attrs["cellmap"] = group_attrs
        crop.attrs.refresh()
        logger.info(f"check new class list: {crop.attrs['cellmap']['annotation']['class_names']}")

def corner_offset(center_off_arr, raw_res_arr, crop_res_arr):
    return (center_off_arr + raw_res_arr/2. - crop_res_arr/2.)

def valid_offset(center_off_arr, raw_res_arr, crop_res_arr):
    corner_off_arr = corner_offset(center_off_arr, raw_res_arr, crop_res_arr)
    return np.all(corner_off_arr % raw_res_arr == 0) and np.all(corner_off_arr % crop_res_arr == 0)

@cli.command()
@click.argument("crop_path", type=str)
@click.option("--orig_offset", type=float, nargs=3)   
@click.option("--new_offset", type=float, nargs=3) 
@click.option("-r", "--round", is_flag=True)
def edit_offset(crop_path, orig_offset, new_offset, round: bool=False):
    edit_offset_func(crop_path, orig_offset, new_offset, round=round)

def edit_offset_func(crop_path, orig_offset, new_offset, round: bool=False):
    diff_offset = np.array(orig_offset) - np.array(new_offset)
    crop_root = fst.access(crop_path, mode="r+")
    for lbl in list(crop_root.keys()):
        crop_label_root = crop_root[lbl]
        ms_attrs = crop_label_root.attrs["multiscales"]
        for ds in range(len(ms_attrs[0]["datasets"])):
            for k in range(len(ms_attrs[0]["datasets"][ds]["coordinateTransformations"])):
                if ms_attrs[0]["datasets"][ds]["coordinateTransformations"][k]["type"] == "translation":
                    transl = ms_attrs[0]["datasets"][ds]["coordinateTransformations"][k]["translation"] 
                    set_offset = np.array(transl) - diff_offset
                    if round:
                        set_offset = np.round(set_offset)    
                    ms_attrs[0]["datasets"][ds]["coordinateTransformations"][k]["translation"] = list(set_offset)
        crop_label_root.attrs["multiscales"] = ms_attrs

@cli.command()
@click.argument("data-config", type=click.File("rb"))
@click.option("-n", "--dry-run", is_flag=True)
@click.option("--gt_path", type=str, help="path to groundtruth crops", default="/nrs/saalfeld/heinrichl/data/cellmap_labels/fly_organelles/", show_default=True)
def fix_offset(data_config, gt_path: str = "/nrs/saalfeld/heinrichl/data/cellmap_labels/fly_organelles/", dry_run: bool = False):
    datas = yaml.safe_load(data_config)
    summary = {
        "valid": [], 
        "invalid": []
    }
    if dry_run:
        summary["correctable"] = []
    else:
        summary["corrected"] = []
    for key, ds_info in datas.items():
        logger.info(f"Processing {key}")
        root_raw = fst.read(ds_info["raw"])
        try: 
            datasets = root_raw.attrs["multiscales"][0]["datasets"]
        except KeyError as e:
            logger.info("May have different metadata format. Proceeding with next dataset")
            continue
        for ds in datasets:
            if ds["path"] == "s0":
                raw_res = ds["coordinateTransformations"][0]["scale"]
        raw_res_arr = np.array(raw_res)
        for crop in ds_info["crops"]:
            print(f"{gt_path}{key}/groundtruth.zarr/{crop}")
            crop_path = f"{gt_path}{key}/groundtruth.zarr/{crop}"
            crop_root = fst.read(crop_path)
            lbl = list(crop_root.keys())[0]
            crop_datasets = crop_root[lbl].attrs["multiscales"][0]["datasets"]
            for cds in crop_datasets:
                if cds["path"] == "s0":
                    crop_res = cds["coordinateTransformations"][0]["scale"]
                    crop_res_arr = np.array(crop_res)
                    crop_off = cds["coordinateTransformations"][1]["translation"]
                    crop_off_arr = np.array(crop_off)
                    
                    if valid_offset(crop_off_arr, raw_res_arr, crop_res_arr):
                        logger.info(f"Crop {crop} passed with raw resolution: {raw_res}, crop resolution {crop_res} and crop offset {crop_off}")
                        summary["valid"].append(crop)
                    else:
                        logger.info(f"Crop {crop} did not pass with raw resolution: {raw_res}, crop resolution {crop_res} and crop offset {crop_off}")
                        if np.all(crop_res < raw_res):
                            crop_off_arr_suggested = crop_off_arr - crop_res_arr/2.
                            
                        else:
                            crop_off_arr_suggested = crop_off_arr + crop_res_arr /2. - raw_res_arr / 2.
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

