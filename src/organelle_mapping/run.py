import logging
import multiprocessing
from pathlib import Path

import click
import gunpowder as gp
import torch
import yaml
from pydantic import TypeAdapter

from organelle_mapping.checkpoint_edit import create_transfer_checkpoint
from organelle_mapping.config import RunConfig
from organelle_mapping.train import make_train_pipeline
from organelle_mapping.utils import setup_package_logger

logger = logging.getLogger(__name__)


def run(run: RunConfig):
    # Handle finetuning checkpoint preparation if configured
    if run.finetuning is not None:
        checkpoint_path = Path(run.finetuning.source_checkpoint.name)
        # Only prepare if checkpoint doesn't exist (for resumability)
        if not checkpoint_path.exists():
            logger.info(f"Preparing finetuning checkpoint from {run.finetuning.source_experiment}")
            create_transfer_checkpoint(run.finetuning)
        else:
            logger.info(f"Checkpoint {checkpoint_path} already exists, skipping preparation")

    voxel_size = list(run.sampling.values())
    input_size = gp.Coordinate(run.architecture.input_shape) * gp.Coordinate(voxel_size)
    output_size = gp.Coordinate(run.architecture.output_shape) * gp.Coordinate(voxel_size)

    pipeline = make_train_pipeline(run, input_size, output_size)

    with gp.build(pipeline) as pp:
        for _i in range(run.iterations):
            request = gp.BatchRequest()
            request.add(gp.ArrayKey("OUTPUT"), output_size, voxel_size=gp.Coordinate(voxel_size))
            request.add(gp.ArrayKey("RAW"), input_size, voxel_size=gp.Coordinate(voxel_size))
            request.add(gp.ArrayKey("TARGETS"), output_size, voxel_size=gp.Coordinate(voxel_size))
            request.add(gp.ArrayKey("MASK"), output_size, voxel_size=gp.Coordinate(voxel_size))
            if run.min_valid_fraction > 0:
                request.add(gp.ArrayKey("VALID"), output_size, voxel_size=gp.Coordinate(voxel_size))
            pp.request_batch(request)


@click.command()
@click.argument("run-config", type=click.File("rb"))
@click.option(
    "--log-level",
    "log_levels",
    multiple=True,
    default=("INFO",),
    help=(
        "Logging level. Use 'LEVEL' (e.g. 'DEBUG') to set the organelle_mapping logger, "
        "or '<logger>.<LEVEL>' (e.g. 'gunpowder.DEBUG', 'lsd_lite.ERROR') for other loggers. "
        "May be passed multiple times."
    ),
)
def main(run_config, log_levels=("INFO",)):
    # Force fork start method for gunpowder PreCache workers so they inherit
    # the LSD service singleton (a module-level global in corditea._lsd_service).
    # Python 3.14 made forkserver the default on Linux, which breaks that
    # inheritance and causes workers to fall back to in-process JAX, hitting
    # CUDA_ERROR_DEVICE_UNAVAILABLE en masse against the device the service holds.
    multiprocessing.set_start_method("fork", force=True)

    # Pre-init main's CUDA on GPU 0 before any subprocess spawn. On clusters
    # with Exclusive_Process compute mode, JAX's startup in the LSD service
    # process can transiently touch GPU 0 during device enumeration and grab
    # the exclusive lock; main's subsequent model.to('cuda') then fails with
    # "CUDA-capable device(s) is/are busy or unavailable". Forcing the lock
    # in main first makes the service's enumeration cleanly bounce off.
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")

    setup_package_logger(log_levels)

    config = TypeAdapter(RunConfig).validate_python(
        yaml.safe_load(run_config), context={"base_dir": Path(run_config.name).parent}
    )
    run(config)
