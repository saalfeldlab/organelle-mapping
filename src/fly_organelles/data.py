import gunpowder as gp
import fibsem_tools as fst
import logging
from pathlib import Path
import xarray as xr
import numpy as np
import corditea
import math
import yaml
import os
from fly_organelles.utils import corner_offset, find_target_scale, get_scale_info, find_target_scale_by_offset
from typing import BinaryIO

logger = logging.getLogger(__name__)
                
def spatial_spec_from_xarray(xarr) -> tuple[gp.Roi, gp.Coordinate]:
    assert isinstance(xarr, xr.DataArray)
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
    ):
        super().__init__()
        self.stores = {}
        raw_grp = fst.read(raw_store)
        raw_scale, raw_offset, raw_shape = find_target_scale(raw_grp, sampling)
        self.stores[raw_arraykey] = fst.read(Path(raw_store) / raw_scale)
        self.labels = labels
        raw_roi = gp.Roi(raw_offset, gp.Coordinate(sampling) * gp.Coordinate(raw_shape))
        raw_voxel_size = gp.Coordinate(sampling)
        # raw_roi, raw_voxel_size = spatial_spec_from_xarray(self.stores[raw_arraykey])
        self.specs[raw_arraykey] = gp.array_spec.ArraySpec(
            roi=raw_roi, voxel_size=raw_voxel_size, interpolatable=True, dtype=self.stores[raw_arraykey].dtype
        )
        for label, labelkey in self.labels.items():
            label_grp = fst.read(Path(label_store) / label)
            label_scale, label_offset, label_shape = find_target_scale(label_grp, sampling)
            self.stores[labelkey] = fst.read_xarray(Path(label_store) / label / label_scale)
            # label_roi, label_voxel_size = spatial_spec_from_xarray(self.stores[labelkey])
            label_roi = gp.Roi(label_offset, gp.Coordinate(label_shape)*gp.Coordinate(sampling))
            label_voxel_size = gp.Coordinate(sampling)
            self.specs[labelkey] = gp.array_spec.ArraySpec(
                roi=label_roi, voxel_size=label_voxel_size, interpolatable=False, dtype=self.stores[labelkey].dtype
            )
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
            vs = self.specs[ak].voxel_size
            dataset_roi = rs.roi / vs
            dataset_roi = dataset_roi - self.spec[ak].roi.offset / vs
            # loc = {axis:slice(b, e, None) for b, e, axis in zip(rs.roi.get_begin(), rs.roi.get_end()-vs/2., "zyx")}
            # arr = self.stores[ak].sel(loc).to_numpy()
            arr = np.asarray(self.stores[ak][dataset_roi.to_slices()])
            array_spec = self.specs[ak].copy()
            array_spec.roi = rs.roi
            batch.arrays[ak] = gp.Array(arr, array_spec)
        logger.debug("done")
        timing.stop()
        batch.profiling_stats.add(timing)
        return batch


class ExtractMask(gp.BatchFilter):
    def __init__(self, label_key, mask_key, mask_value=255):
        super().__init__()
        self.mask_key = mask_key
        self.mask_value = 255
        self.label_key = label_key

    def setup(self):
        assert self.label_key in self.spec, f"Need {self.label_key}"
        spec = self.spec[self.label_key].copy()
        self.provides(self.mask_key, spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.label_key] = request[self.mask_key].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()
        label_arr = batch[self.label_key].data
        mask = label_arr != self.mask_value
        spec = self.spec[self.label_key].copy()
        spec.roi = request[self.label_key].roi
        outputs.arrays[self.mask_key] = gp.Array(mask.astype(spec.dtype), spec)
        return outputs
