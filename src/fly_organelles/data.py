import logging
from pathlib import Path

import fibsem_tools as fst
import gunpowder as gp
import numpy as np
import xarray as xr

import fly_organelles.utils as utils

logger = logging.getLogger(__name__)


class CellMapCropSource(gp.batch_provider.BatchProvider):
    def __init__(
        self,
        label_store: str,
        raw_store: str,
        labels: dict[str, gp.array.ArrayKey],
        raw_arraykey: gp.ArrayKey,
        sampling: dict[str, int],
        base_padding: gp.Coordinate,
        max_request: gp.Coordinate,
    ):
        super().__init__()
        self.stores = {}
        self.specs = {}
        self.max_request = max_request
        raw_grp = fst.read(raw_store)
        raw_axes_names = utils.get_axes_names(raw_grp)
        self.axes_order = raw_axes_names
        raw_scale, _, raw_res, raw_shape = utils.find_target_scale(raw_grp, sampling, 0)
        raw_native_scale = utils.get_scale_info(raw_grp, 0)[1]["s0"]
        raw_corner_offset = gp.Coordinate(
            (0,) * len(sampling)
        )  # raw corner-offset is 0 by definition
        self.labels = labels
        self.needs_downsampling = None
        raw_voxel_size = gp.Coordinate(utils.ax_dict_to_list(sampling, raw_axes_names))
        # raw_roi, raw_voxel_size = spatial_spec_from_xarray(self.stores[raw_arraykey])

        self.padding = base_padding
        for label, labelkey in self.labels.items():
            label_grp = fst.read(Path(label_store) / label)
            label_axes_names = utils.get_axes_names(label_grp)
            label_scale, label_offset, _, label_shape = utils.find_target_scale(
                label_grp, sampling
            )
            label_corner_offset = gp.Coordinate(
                utils.undecimal_arr(
                    utils.corner_offset(
                        utils.decimal_arr(
                            utils.ax_dict_to_list(label_offset, label_axes_names)
                        ),
                        utils.decimal_arr(
                            utils.ax_dict_to_list(raw_native_scale, raw_axes_names)
                        ),
                        utils.decimal_arr(
                            utils.ax_dict_to_list(raw_res, raw_axes_names)
                        ),
                    )
                )
            )
            # label_offset = tuple(np.array(label_offset) - np.array(sampling)/2.)
            self.stores[labelkey] = fst.read_xarray(
                Path(label_store) / label / label_scale
            )
            # label_roi, label_voxel_size = spatial_spec_from_xarray(self.stores[labelkey])
            label_voxel_size = raw_voxel_size
            cropsize = (
                gp.Coordinate(utils.ax_dict_to_list(label_shape, label_axes_names))
                * label_voxel_size
            )
            # this is a trick to avoid gunpowder problem of offsets needing to be divisible by the sampling
            self.secret_raw_offset = label_corner_offset % raw_voxel_size
            label_corner_offset -= self.secret_raw_offset
            label_roi = gp.Roi(label_corner_offset, cropsize)

            self.specs[labelkey] = gp.array_spec.ArraySpec(
                roi=label_roi,
                voxel_size=label_voxel_size,
                interpolatable=False,
                dtype=self.stores[labelkey].dtype,
            )

            if self.secret_raw_offset != gp.Coordinate((0,) * len(sampling)):
                if self.needs_downsampling:
                    msg = f"Inconsistent offsets in labels of {label_store}"
                    raise ValueError(msg)
                self.needs_downsampling = False
            else:
                if self.needs_downsampling is not None and not self.needs_downsampling:
                    msg = f"Inconsistent offsets in labels of {label_store}"
                    raise ValueError(msg)
                self.needs_downsampling = True
        if self.needs_downsampling:
            logger.debug(
                f"Need to downsample raw for {label_store} to accomodate offset {label_corner_offset}."
            )
            raw_up_scale, _, raw_up_res, raw_up_shape = (
                utils.find_target_scale_by_offset(raw_grp, utils.list_to_ax_dict(label_corner_offset, label_axes_names))
            )
            logger.debug(
                f"Reading raw from {raw_store}/ {raw_up_scale} with voxel_size {raw_up_res}"
            )
            sampling_up = (raw_up_res / raw_res) * sampling
            # TODO make a drawing for this
            raw_roi = gp.Roi(
                raw_corner_offset,
                gp.Coordinate(utils.ax_dict_to_list(raw_up_shape, raw_axes_names))
                * gp.Coordinate(utils.ax_dict_to_list(sampling_up, raw_axes_names))
                - raw_voxel_size,
            )
            self.stores[raw_arraykey] = fst.read(Path(raw_store) / raw_up_scale)
            raw_spec = gp.array_spec.ArraySpec(
                roi=raw_roi,
                voxel_size=raw_up_res,
                interpolatable=True,
                dtype=self.stores[raw_arraykey].dtype,
            )

        else:
            self.secret_raw_offset = gp.Coordinate((0,) * len(sampling))
            raw_roi = gp.Roi(
                raw_corner_offset,
                gp.Coordinate(utils.ax_dict_to_list(sampling, raw_axes_names))
                * gp.Coordinate(utils.ax_dict_to_list(raw_shape, raw_axes_names)),
            )
            self.stores[raw_arraykey] = fst.read(Path(raw_store) / raw_scale)

            raw_spec = gp.array_spec.ArraySpec(
                roi=raw_roi,
                voxel_size=raw_voxel_size,
                interpolatable=True,
                dtype=self.stores[raw_arraykey].dtype,
            )

            self.raw_arraykey = raw_arraykey

        self.padding += (
            gp.Coordinate(
                max(0, p) for p in self.max_request - (cropsize + self.padding * 2)
            )
            / 2.0
        )
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
                logger.debug(
                    f"Shifting {ak} dataset_roi by secret raw offset {dataset_roi}"
                )
            else:
                dataset_roi = rs.roi
            dataset_roi = dataset_roi / vs
            dataset_roi = dataset_roi - self.spec[ak].roi.offset / vs
            logger.debug(
                f"Reading {ak} with dataset_roi {dataset_roi} ({dataset_roi.to_slices()})"
            )
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


class ExtractMask(gp.BatchFilter):
    def __init__(self, label_key, mask_key, mask_value=255):
        super().__init__()
        self.mask_key = mask_key
        self.mask_value = mask_value
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
