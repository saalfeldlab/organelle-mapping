import itertools
import logging
import os
from decimal import Decimal
from statistics import mode
from typing import BinaryIO, Literal, Optional

import gunpowder as gp
import numpy as np
import yaml
import zarr
from cellmap_schemas.annotation import (
    AnnotationArrayAttrs,
    InstancePossibility,
    SemanticPossibility,
    SemanticSegmentation,
)
from pydantic_ome_ngff.v04.axis import Axis
from pydantic_ome_ngff.v04.multiscale import (
    Dataset,
    MultiscaleGroupAttrs,
    MultiscaleMetadata,
)
from pydantic_ome_ngff.v04.transform import VectorScale, VectorTranslation

logger = logging.getLogger(__name__)


def setup_package_logger(log_level: str = "INFO") -> None:
    """Set up the organelle_mapping package logger with consistent formatting.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    pkg_logger = logging.getLogger("organelle_mapping")
    pkg_logger.setLevel(log_level.upper())

    # Add console handler if none exists
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        pkg_logger.addHandler(handler)
        pkg_logger.propagate = False  # Don't propagate to root logger


def decimal_arr(arr, precision: int = 2) -> np.ndarray:
    return np.array([Decimal(f"{a:.{precision}f}") for a in arr])


def undecimal_arr(arr, precision: int = 2) -> np.ndarray:
    return np.array([float(round(a, precision)) for a in arr])


def corner_offset(center_off_arr, raw_res_arr, crop_res_arr):
    return center_off_arr + raw_res_arr / Decimal("2.0") - crop_res_arr / Decimal("2")


def valid_offset(center_off_arr, raw_res_arr, crop_res_arr):
    corner_off_arr = corner_offset(decimal_arr(center_off_arr), decimal_arr(raw_res_arr), decimal_arr(crop_res_arr))
    # for co, rr, cr in zip(corner_off_arr, raw_res_arr, crop_res_arr):
    #     if not np.isclose(float(Decimal(str(co)) % Decimal(str(rr))),0):
    #         print(co, rr, co % rr)
    #         return False
    #     if not np.isclose(float(Decimal(str(co)) % Decimal(str(cr))),0):
    #         print(co, cr, co % cr)
    #         return False
    # return True
    if not (np.all(corner_off_arr % raw_res_arr == Decimal(0)) and np.all(corner_off_arr % crop_res_arr == Decimal(0))):
        print(
            center_off_arr,
            raw_res_arr,
            crop_res_arr,
            corner_off_arr % raw_res_arr,
            corner_off_arr % crop_res_arr,
        )
    return np.all(corner_off_arr % raw_res_arr == Decimal(0)) and np.all(corner_off_arr % crop_res_arr == Decimal(0))


def all_combinations(iterable):
    for r in range(1, len(iterable) + 1):
        yield from itertools.combinations(iterable, r)


def read_label_yaml(yaml_file: BinaryIO) -> dict[str, set[str]]:
    classes = yaml.safe_load(yaml_file)
    for lbl, atoms in classes.items():
        classes[lbl] = set(atoms)
    return classes


def read_data_yaml(yaml_file: BinaryIO):
    datasets = yaml.safe_load(yaml_file)
    label_stores = []
    raw_stores = []
    crop_copies = []
    for dataset, ds_info in datasets["datasets"].items():
        for crop in ds_info["crops"]:
            copies = crop.split(",")
            for c in copies:
                label_stores.append(os.path.join(datasets["gt_path"], dataset, "groundtruth.zarr", c))
                raw_stores.append(ds_info["raw"])
                crop_copies.append(len(copies))
    return label_stores, raw_stores, crop_copies


def get_axes_names(zarr_grp, multiscale: str | int = 0) -> list[str]:
    if isinstance(multiscale, int):
        index = multiscale
    else:
        for i, ms in enumerate(zarr_grp.attrs["multiscales"]):
            if ms.get("name") == multiscale:
                index = i
                break
        else:
            # raise an error if no matching multiscale found
            msg = f"Multiscale with name '{multiscale}' not found in Zarr group at {zarr_grp.store.path}"
            raise KeyError(msg)
    return [ax["name"] for ax in zarr_grp.attrs["multiscales"][index]["axes"]]


def get_scale_info(
    zarr_grp,
    multiscale: str | int = 0,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    """
    Extracts scale information, including resolutions, offsets, and shapes from a Zarr group.

    Args:
        zarr_grp: A Zarr group object containing multiscale datasets and associated metadata.
        multiscale (str | int): The name or index of the multiscale to extract information from. Default is 0, which refers to the first multiscale specification in the attributes list.
    Returns:
        tuple: A tuple containing the following:
            - offsets (dict[str, dict[str, float]]): A dictionary mapping dataset paths to dictionaries of axis names and their corresponding offsets.
            - resolutions (dict[str, dict[str, float]]): A dictionary mapping dataset paths to dictionaries of axis names and their corresponding resolutions.
            - shapes (dict[str, dict[str, int]]): A dictionary mapping dataset paths to dictionaries of axis names and their corresponding shapes.

    Raises:
        KeyError: If the expected attributes or keys are missing in the Zarr group metadata.

    Notes:
        - This function assumes the Zarr group follows the multiscale metadata specification.
        - The function relies on the presence of "multiscales" and "coordinateTransformations" attributes in the Zarr group metadata.
    """

    attrs = zarr_grp.attrs
    resolutions = {}
    offsets = {}
    shapes = {}
    axes_names = get_axes_names(zarr_grp)
    # making a ton of assumptions here, hopefully triggering KeyErrors though if they don't apply
    if isinstance(multiscale, int):
        index = multiscale
    else:
        for i, ms in enumerate(attrs["multiscales"]):
            if ms.get("name") == multiscale:
                index = i
                break
        else:
            # raise an error if no matching multiscale found
            msg = f"Multiscale with name '{multiscale}' not found in Zarr group at {zarr_grp.store.path}"
            raise KeyError(msg)
    for scale in attrs["multiscales"][index]["datasets"]:
        resolutions[scale["path"]] = dict(zip(axes_names, scale["coordinateTransformations"][0]["scale"]))
        offsets[scale["path"]] = dict(zip(axes_names, scale["coordinateTransformations"][1]["translation"]))
        shapes[scale["path"]] = dict(zip(axes_names, zarr_grp[scale["path"]].shape))
    # offset = min(offsets.values())
    return offsets, resolutions, shapes


def find_target_scale_by_offset(
    zarr_grp,
    target_offset: dict[str, float],
    multiscale: Optional[str | int | tuple[str | int]] = None,
) -> tuple[str, dict[str, float], dict[str, float], dict[str, int]]:
    if multiscale is None:
        multiscale = get_multiscale_names(zarr_grp)
    if isinstance(multiscale, (str, int)):
        multiscale = [multiscale]
    ms_offsets, ms_resolutions, ms_shapes = {}, {}, {}
    for ms in multiscale:
        offsets, resolutions, shapes = get_scale_info(zarr_grp, multiscale=ms)
        if isinstance(ms, int):
            ms_name = get_multiscale_names(zarr_grp)[ms]
        else:
            ms_name = ms
        ms_offsets[ms_name] = offsets
        ms_resolutions[ms_name] = resolutions
        ms_shapes[ms_name] = shapes
    target_scale = None
    target_ms_name = None

    for ms_name, resolutions in ms_resolutions.items():
        for scale, res in reversed(resolutions.items()):
            if all(off % res[ax] == 0 for ax, off in target_offset.items()):
                target_scale = scale
                target_ms_name = ms_name
                break
        if target_scale is not None:
            break
    if target_scale is None:
        msg = f"Zarr {zarr_grp.store.path}, {zarr_grp.path} does not contain array compatible with target offset {target_offset}"
        raise ValueError(msg)
    return (
        target_scale,
        ms_offsets[target_ms_name][target_scale],
        ms_resolutions[target_ms_name][target_scale],
        ms_shapes[target_ms_name][target_scale],
    )


def get_multiscale_names(zarr_grp) -> list[str]:
    """Returns the names of all multiscales in a Zarr group.

    Args:
        zarr_grp: A Zarr group object containing multiscale datasets and associated metadata.

    Returns:
        list[str]: A list of multiscale names.

    Raises:
        KeyError: If the expected attributes or keys are missing in the Zarr group metadata.
    """
    attrs = zarr_grp.attrs
    multiscale_names = []
    for multiscale in attrs["multiscales"]:
        if "name" in multiscale:
            multiscale_names.append(multiscale["name"])
    return multiscale_names


def find_target_scale(
    zarr_grp: zarr.Group,
    target_resolution: dict[str, float],
    multiscale: Optional[str | int | tuple[str | int]] = None,
) -> tuple[str, dict[str, float], dict[str, float], dict[str, float], dict[str, int]]:
    """Finds the scale in a Zarr group that matches the specified target resolution.

    Args:
        zarr_grp (zarr.Group): The Zarr group containing multiscale data.
        target_resolution (dict[str, float]): A dictionary specifying the target resolution for each axis.

    Raises:
        ValueError: If no scale in the Zarr group matches the target resolution.

    Returns:
        tuple[str, dict[str, float], dict[str, float], dict[str, float], dict[str, int]]:
            - The name of the target scale.
            - The offset dictionary for the target scale.
            - The resolution dictionary for the target scale.
            - The shape dictionary for the target scale.

    """
    if multiscale is None:
        multiscale = get_multiscale_names(zarr_grp)
    if isinstance(multiscale, (str, int)):
        multiscale = [multiscale]
    ms_offsets, ms_resolutions, ms_shapes = {}, {}, {}
    for ms in multiscale:
        offsets, resolutions, shapes = get_scale_info(zarr_grp, multiscale=ms)
        if isinstance(ms, int):
            ms_name = get_multiscale_names(zarr_grp)[ms]
        else:
            ms_name = ms
        ms_offsets[ms_name] = offsets
        ms_resolutions[ms_name] = resolutions
        ms_shapes[ms_name] = shapes
    target_scale = None
    target_ms_name = None
    for ms_name, resolutions in ms_resolutions.items():
        for scale, res in resolutions.items():
            if all(np.isclose(res[ax], target_resolution[ax]) for ax in target_resolution):
                target_scale = scale
                target_ms_name = ms_name
                break
        if target_scale is not None:
            break
    if target_scale is None:
        msg = f"Zarr {zarr_grp.store.path}, {zarr_grp.path} does not contain array compatible with target resolution {target_resolution}"
        raise ValueError(msg)
    return (
        target_scale,
        ms_offsets[target_ms_name][target_scale],
        ms_resolutions[target_ms_name][target_scale],
        ms_shapes[target_ms_name][target_scale],
    )


class SmoothSemanticSegmentation(SemanticSegmentation):
    """
    Metadata for a semantic segmentation, i.e. a segmentation where unique
    numerical values represent separate semantic classes.

    Attributes
    ----------

    type: Literal["semantic_segmentation"]
        Must be the string 'semantic_segmentation'.
    encoding: dict[SemanticPossibility, int]
        This dict represents the mapping from possibilities to numeric values. The keys
        must be strings in the set `{'unknown', 'absent', 'present'}`, and the values
        must be numeric values contained in the array described by this metadata.

        For example, if an annotator produces an array where 0 represents `unknown` and
        1 represents the presence of class X then `encoding` would take the value
        `{'unknown': 0, 'present': 1}`

    """

    type: Literal["smooth_semantic_segmentation"] = "smooth_semantic_segmentation"


class GeneralizedAnnotationArrayAttrs(AnnotationArrayAttrs):
    complement_counts: dict[InstancePossibility, int] | dict[SemanticPossibility, int | float] | None


def generate_standard_multiscale(dataset_paths, axes, base_resolution, base_offset, factors, name="nominal"):
    axes_order = [ax.name for ax in axes]
    scale = np.array(ax_dict_to_list(base_resolution, axes_order))
    offset = np.array(ax_dict_to_list(base_offset, axes_order))
    transforms = [(VectorScale(scale=tuple(scale)), VectorTranslation(translation=tuple(offset)))]
    for f in factors:
        factor = np.array(ax_dict_to_list(f, axes_order))
        offset = offset + (factor - np.ones_like(factor)) * scale / 2.0
     >   scale = scale * factor
        transforms.append(
            (
                VectorScale(scale=tuple(scale)),
                VectorTranslation(translation=tuple(offset)),
            )
        )
    datasets = tuple(
        Dataset(path=dataset_path, coordinateTransformations=transform)
        for dataset_path, transform in zip(dataset_paths, transforms)
    )
    # axes = tuple(Axis(name=ax, type="space", unit="nanometer") for ax in ("z", "y", "x"))
    return MultiscaleMetadata(name=name, axes=axes, type=None, datasets=datasets)


def get_axes_object(zarr_grp, multiscale_name: Optional[str] = None) -> list[Axis]:
    if multiscale_name is None:
        index = 0
    else:
        for index, multiscale in enumerate(zarr_grp.attrs["multiscales"]):
            if multiscale.get("name") == multiscale_name:
                break
        else:
            # raise an error if no matching multiscale found
            msg = f"Multiscale with name '{multiscale_name}' not found in Zarr group at {zarr_grp.store.path}"
            raise KeyError(msg)
    msattrs = MultiscaleGroupAttrs(multiscales=zarr_grp.attrs["multiscales"])
    return msattrs.multiscales[index].axes


def infer_nominal_transform(scale: dict[str, float], offset: dict[str, float]) -> tuple[dict[str, int], dict[str, int]]:
    if offset.keys() != scale.keys():
        msg = f"Keys of offset {offset.keys()} do not match keys of scale {scale.keys()}."
    if len(set(scale.values())) == len(scale.values()):
        msg = "Scales along all axes are unique, cannot infer nominal transform."
        raise ValueError(msg)
    # get dominant scale, i.e. the one that's represented more than once with np unique
    nominal_scale_val = mode(scale.values())
    if int(nominal_scale_val) != nominal_scale_val:
        msg = f"Dominant scale {nominal_scale_val} is not an integer, cannot infer nominal transform."
        raise ValueError(msg)
    nominal_scale_val = int(nominal_scale_val)
    nominal_scale = dict.fromkeys(scale.keys(), nominal_scale_val)
    nominal_offset = {}
    for ax, off in offset.items():
        pix_off = off / scale[ax]
        # assert np.isclose(int(pix_off * 2), pix_off * 2, 1e-4), f"{pix_off}"
        nominal_offset[ax] = int(round(pix_off * nominal_scale_val))
    return nominal_scale, nominal_offset


def ax_dict_to_list(ax_dict, axes_order):
    return [ax_dict[ax] for ax in axes_order]


def list_to_ax_dict(iterable, axes_order):
    return dict(zip(axes_order, iterable))


def get_downsampling_factors(samplings):
    sorted_keys = sorted(samplings.keys(), key=lambda x: min(samplings[x].values()))
    factors = []
    for sc1, sc2 in itertools.pairwise(sorted_keys):
        factor = {}
        for ax in samplings[sc1].keys():
            factor[ax] = samplings[sc2][ax] / samplings[sc1][ax]
        factors.append(factor)
    return factors
