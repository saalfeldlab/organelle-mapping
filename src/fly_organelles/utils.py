import itertools
import os
from typing import BinaryIO

import gunpowder as gp
import numpy as np
import yaml


def corner_offset(center_off_arr, raw_res_arr, crop_res_arr):
    return center_off_arr + raw_res_arr / 2.0 - crop_res_arr / 2.0


def valid_offset(center_off_arr, raw_res_arr, crop_res_arr):
    corner_off_arr = corner_offset(center_off_arr, raw_res_arr, crop_res_arr)
    return np.all(corner_off_arr % raw_res_arr == 0) and np.all(corner_off_arr % crop_res_arr == 0)


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


def get_scale_info(zarr_grp):
    attrs = zarr_grp.attrs
    resolutions = {}
    offsets = {}
    shapes = {}
    # making a ton of assumptions here, hopefully triggering KeyErrors though if they don't apply
    for scale in attrs["multiscales"][0]["datasets"]:
        resolutions[scale["path"]] = scale["coordinateTransformations"][0]["scale"]
        offsets[scale["path"]] = scale["coordinateTransformations"][1]["translation"]
        shapes[scale["path"]] = zarr_grp[scale["path"]].shape
    # offset = min(offsets.values())
    return offsets, resolutions, shapes


def find_target_scale_by_offset(zarr_grp, offset):
    offsets, resolutions, shapes = get_scale_info(zarr_grp)
    target_scale = None
    for scale, res in list(resolutions.items())[::-1]:
        if gp.Coordinate(offset) % gp.Coordinate(res) == gp.Coordinate((0,) * len(offset)):
            target_scale = scale
            break
    if target_scale is None:
        msg = f"Zarr {zarr_grp.store.path}, {zarr_grp.path} does not contain array compatible with offset {offset}"
        raise ValueError(msg)
    return target_scale, offsets[target_scale], resolutions[target_scale], shapes[target_scale]


def find_target_scale(zarr_grp, target_resolution):
    offsets, resolutions, shapes = get_scale_info(zarr_grp)
    target_scale = None
    for scale, res in resolutions.items():
        if gp.Coordinate(res) == gp.Coordinate(target_resolution):
            target_scale = scale
            break
    if target_scale is None:
        msg = f"Zarr {zarr_grp.store.path}, {zarr_grp.path} does not contain array with sampling {target_resolution}"
        raise ValueError(msg)
    return target_scale, offsets[target_scale], shapes[target_scale]
