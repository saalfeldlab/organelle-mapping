import pytest
from fly_organelles.data import CellMapCropSource, data_pipeline
import gunpowder as gp
import matplotlib.pyplot as plt
import logging
log = logging.getLogger("gunpowder")
log.setLevel(logging.DEBUG)
class TestDataFlow:
    def test_data_src(self):
        raw = gp.array.ArrayKey("RAW")
        mito_mem = gp.array.ArrayKey("MITO_MEM")
        mito_lum = gp.array.ArrayKey("MITO_LUM")
        src = CellMapCropSource(
            "/nrs/saalfeld/heinrichl/data/cellmap_labels/fly_organelles/jrc_mb-1a/groundtruth.zarr/crop120",
            "/nrs/cellmap/data/jrc_mb-1a/jrc_mb-1a.zarr/recon-1/em/fibsem-uint8",
            {"mito_mem":mito_mem, "mito_lum": mito_lum},
            raw,
            (4,4,4)
            )
        assert isinstance(src,CellMapCropSource)
        test_request = gp.batch_request.BatchRequest()
        test_request[raw] = gp.array_spec.ArraySpec(roi =gp.roi.Roi((19052, 19820, 26000), (700,700,700)))
        test_request[mito_mem] = gp.array_spec.ArraySpec(roi =gp.roi.Roi((19052, 19820, 26000), (700,700,700)))
        test_request[mito_lum] = gp.array_spec.ArraySpec(roi =gp.roi.Roi((19052, 19820, 26000), (700,700,700)))
        with gp.build(src):
            batch = src.request_batch(test_request)
        assert batch[raw].data.shape == (175,175,175)
        assert batch[mito_mem].data.shape == (175,175,175)
        assert batch[mito_lum].data.shape == (175,175,175)
        assert src.get_size() == 200*200*200
        # plt.imsave("raw.png", batch[raw].data[0], cmap="Greys_r")
        # plt.imsave("mito_mem.png", batch[mito_mem].data[0], vmin=0,vmax=1)
        # plt.imsave("mito_lum.png", batch[mito_lum].data[0], vmin=0,vmax=1)
    def test_data_pipeline(self):
        pipeline = data_pipeline(
            ["mito_mem", "mito_lum"],
            ["/nrs/saalfeld/heinrichl/data/cellmap_labels/fly_organelles/jrc_mb-1a/groundtruth.zarr/crop120",],
            ["/nrs/cellmap/data/jrc_mb-1a/jrc_mb-1a.zarr/recon-1/em/fibsem-uint8"],
            gp.Coordinate((40,40,40)),
            (4,4,4)
        )
        test_request = gp.BatchRequest()
        test_request[gp.ArrayKey("RAW")] = gp.ArraySpec(roi=gp.Roi((0,0,0), (800,800,800)))
        test_request[gp.ArrayKey("MITO_MEM")] = gp.ArraySpec(roi=gp.Roi((0,0,0), (800,800,800)))
        test_request[gp.ArrayKey("MITO_LUM")] = gp.ArraySpec(roi=gp.Roi((0,0,0), (800,800,800)))
        with gp.build(pipeline) as pp:
            batch = pp.request_batch(test_request)
            assert batch[gp.ArrayKey("RAW")].data.shape == (200,200,200)
            assert batch[gp.ArrayKey("MITO_MEM")].data.shape == (200,200,200)
            assert batch[gp.ArrayKey("MITO_LUM")].data.shape == (200,200,200)



