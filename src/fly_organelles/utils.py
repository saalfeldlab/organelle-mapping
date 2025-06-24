import itertools
import os
from typing import BinaryIO
from funlib.geometry import Coordinate
from scipy.ndimage.morphology import distance_transform_edt
import edt
import gunpowder as gp
import numpy as np
import yaml

import logging

logger = logging.getLogger(__name__)

def corner_offset(center_off_arr, raw_res_arr, crop_res_arr):
    return np.round((center_off_arr + raw_res_arr / 2.0 - crop_res_arr / 2.0)/crop_res_arr)*crop_res_arr

# def corner_offset(center_off_arr, raw_res_arr, crop_res_arr):
#     return center_off_arr + raw_res_arr / 2.0 - crop_res_arr / 2.0


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


from gunpowder.nodes import BatchFilter

class ShiftNorm(BatchFilter):
    def __init__(self, array, min, max):
        self.array = array
        self.min = min
        self.max = max

    def process(self, batch, request):
        if self.array not in batch.arrays:
            return

        raw = batch.arrays[self.array]
        raw.data = raw.data.clip(self.min, self.max)-self.min
        raw.data /= (self.max-self.min)
        # print(f"new min: {raw.data.min()}, new max: {raw.data.max()}")

from edt import edt
class Distance(gp.BatchFilter):

  def __init__(self, array,sigma=10.0):
    self.array = array
    self.sigma = sigma

  def setup(self):
    self.spec[self.array].dtype = np.float32

  def process(self, batch, request):

    data = batch[self.array].data
    data = np.tanh((edt(data) - edt(data == 0)) / self.sigma)
    batch[self.array].data = data.astype(np.float32)
    
class Binarize(BatchFilter):
    def __init__(self, array):
        self.array = array

    def process(self, batch, request):
        if self.array not in batch.arrays:
            return

        raw = batch.arrays[self.array]
        data = raw.data
        data[data > 0] = 1
        data[data < 1] = 0
        raw.data = data


class Distances(BatchFilter):
    def __init__(self, array,
                 norm = "tanh",
                 dt_scale_factor = 80.0):
        self.array = array
        self.norm = norm
        self.dt_scale_factor = dt_scale_factor

    def process(self, batch, request):
        if self.array not in batch.arrays:
            return
        
        voxel_size = self.spec[self.array].voxel_size

        raw = batch.arrays[self.array]
        if raw.data.any():
            logger.debug(f"Computing distances for {raw.data.shape} with voxel size {voxel_size}")

            distances = self.compute_distance(raw.data, voxel_size, self.norm, self.dt_scale_factor)
            raw.data = distances[0]

    def compute_distance(
        self,
        labels: np.ndarray,
        voxel_size: Coordinate,
        normalize=None,
        normalize_args=None,
    ):
        """
        Process the labels array and convert it to one-hot encoding.

        Args:
            labels (np.ndarray): The labels array.
            voxel_size (Coordinate): The voxel size.
            normalize (str): The normalization method.
            normalize_args: The normalization arguments.
        Returns:
            np.ndarray: The distances array.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.process(labels, voxel_size, normalize, normalize_args)

        """

        num_dims = len(labels.shape)
        if num_dims == voxel_size.dims:
            channel_dim = False
        elif num_dims == voxel_size.dims + 1:
            channel_dim = True
        else:
            raise ValueError("Cannot handle multiple channel dims")

        if not channel_dim:
            labels = labels[np.newaxis]

        all_distances = np.zeros(labels.shape, dtype=np.float32) - 1

        for ii, channel in enumerate(labels):
            boundaries = self.__find_boundaries(channel)

            # mark boundaries with 0 (not 1)
            boundaries = 1.0 - boundaries

            if np.sum(boundaries == 0) == 0:
                max_distance = min(
                    dim * vs / 2 for dim, vs in zip(channel.shape, voxel_size)
                )
                if np.sum(channel) == 0:
                    distances = -np.ones(channel.shape, dtype=np.float32) * max_distance
                else:
                    distances = np.ones(channel.shape, dtype=np.float32) * max_distance
            else:
                # get distances (voxel_size/2 because image is doubled)
                sampling = tuple(float(v) / 2 for v in voxel_size)
                # fixing the sampling for 2D images
                if len(boundaries.shape) < len(sampling):
                    sampling = sampling[-len(boundaries.shape) :]
                distances = distance_transform_edt(boundaries, sampling=sampling)
                distances = distances.astype(np.float32)

                # restore original shape
                downsample = (slice(None, None, 2),) * distances.ndim
                distances = distances[downsample]

                # todo: inverted distance
                distances[channel == 0] = -distances[channel == 0]

            if normalize is not None:
                distances = self.__normalize(distances, normalize, normalize_args)

            all_distances[ii] = distances

        return all_distances

    def __find_boundaries(self, labels: np.ndarray):
        """
        Find the boundaries in the labels.

        Args:
            labels: The labels.
        Returns:
            The boundaries.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.__find_boundaries(labels)

        """
        # labels: 1 1 1 1 0 0 2 2 2 2 3 3       n
        # shift :   1 1 1 1 0 0 2 2 2 2 3       n - 1
        # diff  :   0 0 0 1 0 1 0 0 0 1 0       n - 1
        # bound.: 00000001000100000001000      2n - 1

        if labels.dtype == bool:
            # raise ValueError("Labels should not be bools")
            labels = labels.astype(np.uint8)

        logger.debug(f"computing boundaries for {labels.shape}")

        dims = len(labels.shape)
        in_shape = labels.shape
        out_shape = tuple(2 * s - 1 for s in in_shape)

        boundaries = np.zeros(out_shape, dtype=bool)

        logger.debug(f"boundaries shape is {boundaries.shape}")

        for d in range(dims):
            logger.debug(f"processing dimension {d}")

            shift_p = [slice(None)] * dims
            shift_p[d] = slice(1, in_shape[d])

            shift_n = [slice(None)] * dims
            shift_n[d] = slice(0, in_shape[d] - 1)

            diff = (labels[tuple(shift_p)] - labels[tuple(shift_n)]) != 0

            logger.debug(f"diff shape is {diff.shape}")

            target = [slice(None, None, 2)] * dims
            target[d] = slice(1, out_shape[d], 2)

            logger.debug(f"target slices are {target}")

            boundaries[tuple(target)] = diff

        return boundaries

    def __normalize(self, distances, norm, normalize_args):
        """
        Normalize the distances.

        Args:
            distances: The distances to normalize.
            norm: The normalization method.
            normalize_args: The normalization arguments.
        Returns:
            The normalized distances.
        Raises:
            ValueError: If the normalization method is not supported.
        Examples:
            >>> predictor.__normalize(distances, norm, normalize_args)

        """
        if norm == "tanh":
            scale = normalize_args
            return np.tanh(distances / scale)
        else:
            raise ValueError("Only tanh is supported for normalization")
