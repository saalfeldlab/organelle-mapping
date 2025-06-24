import logging
from pathlib import Path

import fibsem_tools as fst
import gunpowder as gp
import numpy as np
import xarray as xr


from fly_organelles.utils import (
    corner_offset,
    find_target_scale,
    find_target_scale_by_offset,
    get_scale_info,
)

logger = logging.getLogger(__name__)


def spatial_spec_from_xarray(xarr) -> tuple[gp.Roi, gp.Coordinate]:
    if not isinstance(xarr, xr.DataArray):
        msg = f"Expected input to be `xarray.DataArray`, not {type(xarr)}"
        raise TypeError(msg)
    offset = []
    for axis in "zyx":
        offset.append(int(xarr.coords[axis][0].data))
    offset = gp.Coordinate(offset)
    voxel_size = []
    for axis in "zyx":
        voxel_size.append(int((xarr.coords[axis][1] - xarr.coords[axis][0]).data))
    voxel_size = gp.Coordinate(voxel_size)
    shape = voxel_size * gp.Coordinate(xarr.shape)
    roi = gp.Roi(offset, shape)
    return roi, voxel_size


class CellMapCropSource(gp.batch_provider.BatchProvider):
    def __init__(
        self,
        label_store: str,
        raw_store: str,
        labels: dict[str, gp.array.ArrayKey],
        raw_arraykey: gp.ArrayKey,
        sampling: list[int],
        base_padding: gp.Coordinate,
        max_request: gp.Coordinate,
    ):
        super().__init__()
        self.stores = {}
        self.specs = {}
        self.max_request = max_request
        raw_grp = fst.read(raw_store)
        raw_scale, raw_offset, raw_shape = find_target_scale(raw_grp, sampling)
        raw_offset = gp.Coordinate((0,) * len(sampling))  # tuple(np.array(raw_offset) - np.array(sampling)/2.)
        raw_native_scale = get_scale_info(raw_grp)[1]["s0"]
        self.stores[raw_arraykey] = fst.read(Path(raw_store) / raw_scale)
        self.labels = labels
        raw_roi = gp.Roi(raw_offset, gp.Coordinate(sampling) * gp.Coordinate(raw_shape))
        raw_voxel_size = gp.Coordinate(sampling)
        # raw_roi, raw_voxel_size = spatial_spec_from_xarray(self.stores[raw_arraykey])
        raw_spec = gp.array_spec.ArraySpec(
            roi=raw_roi, voxel_size=raw_voxel_size, interpolatable=True, dtype=self.stores[raw_arraykey].dtype
        )
        self.padding = base_padding
        for label, labelkey in self.labels.items():
            label_grp = fst.read(Path(label_store) / label)
            label_scale, label_offset, label_shape = find_target_scale(label_grp, sampling)
            label_offset = gp.Coordinate(
                corner_offset(np.array(label_offset), np.array(raw_native_scale), np.array(sampling))
            )
            if label_offset % raw_voxel_size == gp.Coordinate((0,) * len(sampling)):
                self.needs_downsampling = False
                self.secret_raw_offset = gp.Coordinate((0,) * len(sampling))
            else:
                self.needs_downsampling = True
                logger.debug(f"Need to downsample raw for {label_store} to accomodate offset {label_offset}.")
                raw_scale, raw_offset, raw_res, raw_shape = find_target_scale_by_offset(raw_grp, label_offset)
                logger.debug(f"Reading raw from {raw_store}/ {raw_scale} with voxel_size {raw_res}")
                self.stores[raw_arraykey] = fst.read(Path(raw_store) / raw_scale)
                raw_roi = gp.Roi(
                    gp.Coordinate((0,) * len(sampling)),
                    gp.Coordinate(raw_shape) * gp.Coordinate(raw_res) - gp.Coordinate(sampling),
                )
                raw_spec = gp.array_spec.ArraySpec(
                    roi=raw_roi, voxel_size=raw_res, interpolatable=True, dtype=self.stores[raw_arraykey].dtype
                )
                self.secret_raw_offset = label_offset % gp.Coordinate(sampling)
                label_offset -= self.secret_raw_offset

            # label_offset = tuple(np.array(label_offset) - np.array(sampling)/2.)
            self.stores[labelkey] = fst.read_xarray(Path(label_store) / label / label_scale)
            # label_roi, label_voxel_size = spatial_spec_from_xarray(self.stores[labelkey])
            cropsize = gp.Coordinate(label_shape) * gp.Coordinate(sampling)
            label_roi = gp.Roi(label_offset, cropsize)

            label_voxel_size = gp.Coordinate(sampling)
            self.specs[labelkey] = gp.array_spec.ArraySpec(
                roi=label_roi, voxel_size=label_voxel_size, interpolatable=False, dtype=self.stores[labelkey].dtype
            )
            self.raw_arraykey = raw_arraykey

        self.padding += gp.Coordinate(max(0, p) for p in self.max_request - (cropsize + self.padding * 2)) / 2.0
        self.specs[raw_arraykey] = raw_spec
        self.sampling = sampling

    def get_size(self):
        label = next(iter(self.labels.values()))
        return self.stores[label].size

    def setup(self):
        for key, spec in self.specs.items():
            self.provides(key, spec)

    def provide(self, request) -> gp.Batch:
        timing = gp.profiling.Timing(self)
        timing.start()
        batch = gp.batch.Batch()
        for ak, rs in request.array_specs.items():
            logger.debug(f"Requesting {ak} with {rs}")
            vs = self.specs[ak].voxel_size

            if ak == self.raw_arraykey:
                dataset_roi = rs.roi + self.secret_raw_offset
                logger.debug(f"Shifting {ak} dataset_roi by secret raw offset {dataset_roi}")
            else:
                dataset_roi = rs.roi
            dataset_roi = dataset_roi / vs
            dataset_roi = dataset_roi - self.spec[ak].roi.offset / vs
            logger.debug(f"Reading {ak} with dataset_roi {dataset_roi} ({dataset_roi.to_slices()})")
            # loc = {axis:slice(b, e, None) for b, e, axis in zip(rs.roi.get_begin(), rs.roi.get_end()-vs/2., "zyx")}
            # arr = self.stores[ak].sel(loc).to_numpy()
            arr = np.asarray(self.stores[ak][dataset_roi.to_slices()])
            logger.debug(f"Read array of shape {arr.shape}")
            array_spec = self.specs[ak].copy()
            array_spec.roi = rs.roi
            batch.arrays[ak] = gp.Array(arr, array_spec)
        timing.stop()
        batch.profiling_stats.add(timing)
        return batch


# class ExtractMask(gp.BatchFilter):
#     def __init__(self, label_key, mask_key, mask_value=255):
#         super().__init__()
#         self.mask_key = mask_key
#         self.mask_value = mask_value
#         self.label_key = label_key

#     def setup(self):
#         assert self.label_key in self.spec, f"Need {self.label_key}"
#         spec = self.spec[self.label_key].copy()
#         self.provides(self.mask_key, spec)

#     def prepare(self, request):
#         deps = gp.BatchRequest()
#         deps[self.label_key] = request[self.mask_key].copy()
#         return deps

#     def process(self, batch, request):
#         outputs = gp.Batch()
#         label_arr = batch[self.label_key].data
#         mask = label_arr != self.mask_value
#         spec = self.spec[self.label_key].copy()
#         spec.roi = request[self.label_key].roi
#         outputs.arrays[self.mask_key] = gp.Array(mask.astype(spec.dtype), spec)
#         return outputs



class ExtractMask(gp.BatchFilter):
    def __init__(self, label_key, mask_key, mask_value=255):
        super().__init__()
        self.mask_key = mask_key
        self.mask_value = mask_value
        self.label_key = label_key
        self.count_bg = 0
        self.count_fg = 0

    def setup(self):
        assert self.label_key in self.spec, f"Need {self.label_key}"
        spec = self.spec[self.label_key].copy()
        spec.dtype = np.float32
        self.provides(self.mask_key, spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.label_key] = request[self.mask_key].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()
        label_arr = batch[self.label_key].data
        mask = (label_arr != self.mask_value).astype(np.float32)
        self.count_bg += np.sum(label_arr == 0)
        self.count_fg += np.sum((label_arr != 0) & (label_arr != self.mask_value))
        weight_fg = self.count_fg / (self.count_fg + self.count_bg)
        weight_bg = self.count_bg / (self.count_fg + self.count_bg)
        logger.debug(f"Count: fg={self.count_fg}, bg={self.count_bg}")
        logger.debug(f"Weighting: fg={weight_fg}, bg={weight_bg}")
        if weight_bg == 0:
            weight_bg = 1.0
        if weight_fg == 0:
            weight_fg = 1.0
        
        mask[label_arr == 0] *= weight_fg
        mask[label_arr != 0] *= weight_bg
        spec = self.spec[self.label_key].copy()
        spec.dtype = np.float32
        spec.roi = request[self.label_key].roi
        outputs.arrays[self.mask_key] = gp.Array(mask.astype(spec.dtype), spec)
        return outputs