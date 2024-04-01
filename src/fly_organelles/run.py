from fly_organelles.model import StandardUnet
from fly_organelles.train import make_train_pipeline
import gunpowder as gp
import logging

# logging.basicConfig(level=logging.DEBUG)
# loggp = logging.getLogger("gunpowder.nodes.pad")
# loggp.setLevel(logging.DEBUG)


def run(labels, label_weights, label_stores, raw_stores):
    model = StandardUnet(len(labels))
    voxel_size = (4, 4, 4)
    pipeline = make_train_pipeline(
        model,
        labels,
        label_weights,
        label_stores,
        raw_stores,
        gp.Coordinate(40, 40, 40) * gp.Coordinate(voxel_size),
        voxel_size,
    )
    input_size = gp.Coordinate((178, 178, 178)) * gp.Coordinate(voxel_size)
    output_size = gp.Coordinate((56, 56, 56)) * gp.Coordinate(voxel_size)
    for i in range(10):
        with gp.build(pipeline) as pp:

            request = gp.BatchRequest()
            request.add(gp.ArrayKey("OUTPUT"), output_size)
            request.add(gp.ArrayKey("RAW"), input_size)
            request.add(gp.ArrayKey("LABELS"), output_size)
            request.add(gp.ArrayKey("MASK"), output_size)
            pp.request_batch(request)


def main1():
    labels = ["pm", "ecs", "golgi_lum", "er_lum", "endo_lum", "ves_lum", "mito_lum", "ne_lum", "lyso_lum"]
    label_weights = [
        1.0 / len(labels),
    ] * len(labels)
    label_stores = [""]
    label_stores = [
        "/nrs/saalfeld/heinrichl/data/cellmap_labels/fly_organelles/jrc_mb-1a/groundtruth.zarr/crop120",
    ]
    raw_stores = [
        "/nrs/cellmap/data/jrc_mb-1a/jrc_mb-1a.zarr/recon-1/em/fibsem-uint8",
    ]
    run(labels, label_weights, label_stores, raw_stores)


if __name__ == "__main__":
    main1()
