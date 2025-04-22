import logging

import click
import gunpowder as gp
import monai
import numpy as np
import yaml
from pydantic import TypeAdapter
from organelle_mapping.train import make_train_pipeline
from organelle_mapping.config import RunConfig
logger = logging.getLogger(__name__)


def run(run: RunConfig):

    voxel_size = list(run.sampling.values())
    input_size = gp.Coordinate(run.architecture.input_shape) * gp.Coordinate(voxel_size)
    output_size = gp.Coordinate(run.architecture.output_shape) * gp.Coordinate(voxel_size)

    pipeline = make_train_pipeline(
        run,
        input_size,
        output_size
    )

    with gp.build(pipeline) as pp:
        for i in range(run.iterations):
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
@click.argument("run-config", type=click.File("rb"))
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    help="Set the logging level.",
)
def main(run_config, log_level="INFO"):
    pkg_logger = logging.getLogger("organelle_mapping")
    pkg_logger.setLevel(log_level.upper())
    config = TypeAdapter(RunConfig).validate_python(yaml.safe_load(run_config))
    run(config)
