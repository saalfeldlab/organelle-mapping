import gunpowder as gp
import fibsem_tools as fst
import logging
from pathlib import Path
import xarray as xr
logger = logging.getLogger(__name__)

def spatial_spec_from_xarray(xarr) -> tuple[gp.Roi, gp.Coordinate]:
    assert isinstance(xarr, xr.DataArray)
    offset = []
    for axis in "zyx":
        offset.append(int(xarr.coords[axis][0].data))
    offset = gp.Coordinate(offset)
    voxel_size = []
    for axis in "zyx":
        voxel_size.append(int((xarr.coords[axis][1]-xarr.coords[axis][0]).data))
    voxel_size = gp.Coordinate(voxel_size)
    shape = voxel_size*gp.Coordinate(xarr.shape)
    roi = gp.Roi(offset, shape)
    return roi, voxel_size

class CellMapCropSource(gp.batch_provider.BatchProvider):
    def __init__(
        self,
        label_store,
        raw_store,
        labels: dict[str,gp.array.ArrayKey],
        raw_arraykey
    ):
        super().__init__()
        self.stores = {}
        self.stores[raw_arraykey] = fst.read_xarray(raw_store)
        self.labels = labels
        self.specs = {}
        raw_roi, raw_voxel_size = spatial_spec_from_xarray(self.stores[raw_arraykey])
        self.specs[raw_arraykey] = gp.array_spec.ArraySpec(roi=raw_roi,
                                                           voxel_size=raw_voxel_size,
                                                           interpolatable=True,
                                                           dtype=self.stores[raw_arraykey].dtype)
        for label, labelkey in self.labels.items():
            self.stores[labelkey] = fst.read_xarray(Path(label_store) / label / "s1")
            label_roi, label_voxel_size = spatial_spec_from_xarray(self.stores[labelkey])
            self.specs[labelkey] = gp.array_spec.ArraySpec(roi = label_roi,
                                                           voxel_size=label_voxel_size,
                                                           interpolatable=False,
                                                           dtype=self.stores[labelkey].dtype)

    def setup(self):
        for key, spec in self.specs.items():
            self.provides(key, spec)

    def provide(self, request) -> gp.Batch:
        timing = gp.profiling.Timing(self)
        timing.start()
        batch = gp.batch.Batch()
        for ak, rs in request.array_specs.items():
            vs = self.specs[ak].voxel_size
            loc = {axis:slice(b, e, None) for b, e, axis in zip(rs.roi.get_begin(), rs.roi.get_end()-vs/2., "zyx")}
            arr = self.stores[ak].sel(loc).to_numpy()
            array_spec = self.specs[ak].copy()
            array_spec.roi = rs.roi
            batch.arrays[ak] = gp.Array(arr, array_spec)
        logger.debug("done")
        timing.stop()
        batch.profiling_stats.add(timing)
        return batch
