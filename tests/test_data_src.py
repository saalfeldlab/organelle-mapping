import logging

import gunpowder as gp
import matplotlib.pyplot as plt
import pytest

from fly_organelles.data import CellMapCropSource  # , data_pipeline, add_augmentation
from fly_organelles.train import make_data_pipeline

loggp = logging.getLogger("gunpowder.nodes.intensity_augment")
loggp.setLevel(logging.DEBUG)
logct = logging.getLogger("corditea.gamma_augment")
logct.setLevel(logging.DEBUG)


class TestDataFlow:
    def test_data_src(self):
        raw = gp.array.ArrayKey("RAW")
        mito_mem = gp.array.ArrayKey("MITO_MEM")
        mito_lum = gp.array.ArrayKey("MITO_LUM")
        src = CellMapCropSource(
            "/nrs/saalfeld/heinrichl/data/cellmap_labels/fly_organelles/jrc_mb-1a/groundtruth.zarr/crop120",
            "/nrs/cellmap/data/jrc_mb-1a/jrc_mb-1a.zarr/recon-1/em/fibsem-uint8",
            {"mito_mem": mito_mem, "mito_lum": mito_lum},
            raw,
            (4, 4, 4),
        )
        assert isinstance(src, CellMapCropSource)
        test_request = gp.batch_request.BatchRequest()
        test_request[raw] = gp.array_spec.ArraySpec(roi=gp.roi.Roi((19052, 19820, 26000), (700, 700, 700)))
        test_request[mito_mem] = gp.array_spec.ArraySpec(roi=gp.roi.Roi((19052, 19820, 26000), (700, 700, 700)))
        test_request[mito_lum] = gp.array_spec.ArraySpec(roi=gp.roi.Roi((19052, 19820, 26000), (700, 700, 700)))
        with gp.build(src):
            batch = src.request_batch(test_request)
        assert batch[raw].data.shape == (175, 175, 175)
        assert batch[mito_mem].data.shape == (175, 175, 175)
        assert batch[mito_lum].data.shape == (175, 175, 175)
        assert src.get_size() == 200 * 200 * 200

        plt.imsave("raw.png", batch[raw].data[0], cmap="Greys_r")
        plt.imsave("mito_mem.png", batch[mito_mem].data[0], vmin=0, vmax=1)
        plt.imsave("mito_lum.png", batch[mito_lum].data[0], vmin=0, vmax=1)

    def test_data_pipeline(self):
        labels = ["pm", "ecs", "golgi_lum", "er_lum", "endo_lum", "ves_lum", "mito_lum", "ne_lum", "lyso_lum"]
        label_stores = [
            "/nrs/saalfeld/heinrichl/data/cellmap_labels/fly_organelles/jrc_mb-1a/groundtruth.zarr/crop120",
        ]
        raw_stores = [
            "/nrs/cellmap/data/jrc_mb-1a/jrc_mb-1a.zarr/recon-1/em/fibsem-uint8",
        ]
        voxel_size = (4, 4, 4)
        pipeline = make_data_pipeline(
            labels,
            label_stores,
            raw_stores,
            gp.Coordinate(40, 40, 40) * gp.Coordinate(voxel_size),
            voxel_size,
        )
        input_size = gp.Coordinate((178, 178, 178)) * gp.Coordinate(voxel_size)
        output_size = gp.Coordinate((56, 56, 56)) * gp.Coordinate(voxel_size)

        with gp.build(pipeline) as pp:
            request = gp.BatchRequest()
            request.add()
            request[gp.ArrayKey("LABELS")] = gp.Roi((0, 0, 0), output_size)
            request[gp.ArrayKey("MASK")] = gp.Roi((0, 0, 0), output_size)
            request[gp.ArrayKey("RAW")] = gp.Roi((0, 0, 0), input_size)
            batch = pp.request_batch(request)
        print(batch[gp.ArrayKey("RAW")].data.shape)
        plt.imsave("raw.png", batch[gp.ArrayKey("RAW")].data[0][0][89], vmin=0, vmax=1)
        plt.imsave("pm.png", batch[gp.ArrayKey("LABELS")].data[0][0])
        # plt.imsave("mito_")


#     def test_data_pipeline(self):
#         pipeline = data_pipeline(
#             ["mito_mem", "mito_lum"],
#             ["/nrs/saalfeld/heinrichl/data/cellmap_labels/fly_organelles/jrc_mb-1a/groundtruth.zarr/crop120",],
#             ["/nrs/cellmap/data/jrc_mb-1a/jrc_mb-1a.zarr/recon-1/em/fibsem-uint8"],
#             gp.Coordinate((40,40,40)),
#             (4,4,4)
#         )
#         test_request = gp.BatchRequest()
#         test_request[gp.ArrayKey("RAW")] = gp.ArraySpec(roi=gp.Roi((0,0,0), (800,800,800)))
#         test_request[gp.ArrayKey("MITO_MEM")] = gp.ArraySpec(roi=gp.Roi((0,0,0), (800,800,800)))
#         test_request[gp.ArrayKey("MITO_LUM")] = gp.ArraySpec(roi=gp.Roi((0,0,0), (800,800,800)))
#         with gp.build(pipeline) as pp:
#             batch = pp.request_batch(test_request)
#             assert batch[gp.ArrayKey("RAW")].data.shape == (200,200,200)
#             assert batch[gp.ArrayKey("MITO_MEM")].data.shape == (200,200,200)
#             assert batch[gp.ArrayKey("MITO_LUM")].data.shape == (200,200,200)
#     def test_augment_pipeline(self):
#         pipeline = data_pipeline(
#             ["mito_mem", "mito_lum"],
#             ["/nrs/saalfeld/heinrichl/data/cellmap_labels/fly_organelles/jrc_mb-1a/groundtruth.zarr/crop120",],
#             ["/nrs/cellmap/data/jrc_mb-1a/jrc_mb-1a.zarr/recon-1/em/fibsem-uint8"],
#             gp.Coordinate((400,400,400)),
#             (4,4,4)
#         )
#         pipeline = add_augmentation(pipeline, gp.ArrayKey("RAW"))
#         test_request = gp.BatchRequest()
#         test_request[gp.ArrayKey("RAW")] = gp.ArraySpec(roi=gp.Roi((0,0,0), (600,600,600)))
#         test_request[gp.ArrayKey("MITO_MEM")] = gp.ArraySpec(roi=gp.Roi((0,0,0), (600,600,600)))
#         test_request[gp.ArrayKey("MITO_LUM")] = gp.ArraySpec(roi=gp.Roi((0,0,0), (600,600,600)))
#         with gp.build(pipeline) as pp:
#             batch = pp.request_batch(test_request)
#             assert batch[gp.ArrayKey("RAW")].data.shape == (150,150,150)
#             assert batch[gp.ArrayKey("MITO_MEM")].data.shape == (150,150,150)
#             assert batch[gp.ArrayKey("MITO_LUM")].data.shape == (150,150,150)
#         plt.imsave("raw.png", batch[gp.ArrayKey("RAW")].data[0], cmap="Greys_r")
#         plt.imsave("mito_mem.png", batch[gp.ArrayKey("MITO_MEM")].data[0], vmin=0,vmax=1)
#         plt.imsave("mito_lum.png", batch[gp.ArrayKey("MITO_LUM")].data[0], vmin=0,vmax=1)
#         logging.info("Hello")
# if __name__ == "__main__":
#     dt = TestDataFlow()
#     dt.test_augment_pipeline()
