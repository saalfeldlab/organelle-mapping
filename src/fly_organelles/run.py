import logging

import click
import gunpowder as gp
import numpy as np
import yaml

from fly_organelles.model import StandardUnet
from fly_organelles.train import make_train_pipeline

logger = logging.getLogger(__name__)
# loggp = logging.getLogger("gunpowder.nodes.pad")
# loggp.setLevel(logging.DEBUG)


def run(iterations, labels, label_weights, datasets):
    model = StandardUnet(len(labels))

    sampling = {
        "x": 8,
        "y": 8,
        "z": 8,
    }
    voxel_size = list(sampling.values())
    input_size = gp.Coordinate((178, 178, 178)) * gp.Coordinate(voxel_size)
    output_size = gp.Coordinate((56, 56, 56)) * gp.Coordinate(voxel_size)
    displacement_sigma = gp.Coordinate((24, 24, 24))
    # max_in_request = gp.Coordinate((np.ceil(np.sqrt(sum(input_size**2))),)*len(input_size)) + displacement_sigma * 6
    max_out_request = (
        gp.Coordinate((np.ceil(np.sqrt(sum(output_size**2))),) * len(output_size))
        + displacement_sigma * 6
    )
    pad_width_out = output_size / 2.0

    pipeline = make_train_pipeline(
        model,
        labels=labels,
        label_weights=label_weights,
        datasets=datasets,
        pad_width_out=pad_width_out,
        sampling=sampling,
        max_out_request=max_out_request,
        displacement_sigma=displacement_sigma,
        input_size=input_size,
        output_size=output_size,
        batch_size=14,
    )

    with gp.build(pipeline) as pp:
        for i in range(iterations):
            request = gp.BatchRequest()
            request.add(
                gp.ArrayKey("OUTPUT"), output_size, voxel_size=gp.Coordinate(voxel_size)
            )
            request.add(
                gp.ArrayKey("RAW"), input_size, voxel_size=gp.Coordinate(voxel_size)
            )
            request.add(
                gp.ArrayKey("LABELS"), output_size, voxel_size=gp.Coordinate(voxel_size)
            )
            request.add(
                gp.ArrayKey("MASK"), output_size, voxel_size=gp.Coordinate(voxel_size)
            )
            pp.request_batch(request)


@click.command()
@click.argument("data-config", type=click.File("rb"))
@click.argument("iterations", type=int)
@click.option(
    "--labels",
    "-l",
    multiple=True,
    type=str,
    help="List of labels to train for.",
    default=["organelle", "all_mem"],
)
@click.option("--label_weights", "-lw", multiple=True, type=float)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    help="Set the logging level.",
)
def main(data_config, iterations, labels, label_weights=None, log_level="INFO"):
    pkg_logger = logging.getLogger("fly_organelles")
    pkg_logger.setLevel(log_level.upper())
    if not label_weights:
        label_weights = [
            1.0 / len(labels),
        ] * len(labels)
    else:
        if len(label_weights) != len(labels):
            msg = (
                f"If label weights are specified ({type(label_weights)}),"
                f"they need to be of the same length as the list of labels ({len(labels)})"
            )
            raise ValueError(msg)
        normalizer = np.sum(label_weights)
        label_weights = [lw / normalizer for lw in label_weights]
    logger.info(
        f"Running training for the following labels:"
        f"{', '.join([f'{lbl} ({lblw:.4f})' for lbl,lblw in zip(labels,label_weights)])}"
    )
    # with open(yaml_file, "r") as data_yaml:
    datasets = yaml.safe_load(data_config)
    # label_stores, raw_stores, crop_copies = read_data_yaml(data_yaml)
    run(iterations, labels, label_weights, datasets)
