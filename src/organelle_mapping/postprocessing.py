"""Postprocessing tools for predictions, including ensembling."""

import logging
from pathlib import Path

import click
import dask.array as da
import numpy as np
import zarr
from dask.distributed import Client

from organelle_mapping.utils import setup_package_logger

logger = logging.getLogger(__name__)

MIN_ENSEMBLE_INPUTS = 2


def ensemble_predictions(
    input_paths: list[tuple[Path, str]],
    output_path: tuple[Path, str],
    *,
    weights: list[float] | None = None,
) -> None:
    """Average multiple prediction volumes into a single output.

    Uses dask for lazy, chunked computation with automatic parallelization.

    Args:
        input_paths: List of (container, dataset) tuples for input predictions
        output_path: (container, dataset) tuple for output
        weights: Optional weights for weighted average. If None, uses equal weights.
    """
    if len(input_paths) < MIN_ENSEMBLE_INPUTS:
        msg = f"Need at least {MIN_ENSEMBLE_INPUTS} input predictions to ensemble"
        raise ValueError(msg)

    if weights is None:
        weights = [1.0 / len(input_paths)] * len(input_paths)
    else:
        if len(weights) != len(input_paths):
            msg = f"Number of weights ({len(weights)}) must match number of inputs ({len(input_paths)})"
            raise ValueError(msg)
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

    # Open all input datasets as dask arrays
    input_arrays = []
    zarr_groups = []
    zarr_arrays = []
    array_path = None  # Will be read from first input's metadata

    for container, dataset in input_paths:
        group = zarr.open_group(container / dataset, mode="r")
        zarr_groups.append(group)

        # Get array path from multiscales metadata (e.g., "0" or "s0")
        if array_path is None:
            multiscales = group.attrs.get("multiscales", [{}])
            datasets = multiscales[0].get("datasets", [{}]) if multiscales else [{}]
            array_path = datasets[0].get("path", "s0") if datasets else "s0"

        zarr_arr = group[array_path]
        zarr_arrays.append(zarr_arr)
        dask_arr = da.from_zarr(zarr_arr)
        input_arrays.append(dask_arr)
        logger.info(f"Opened input: {container / dataset}/{array_path}, shape={dask_arr.shape}, dtype={dask_arr.dtype}")

    # Verify all shapes match
    reference_shape = input_arrays[0].shape
    for i, arr in enumerate(input_arrays[1:], 1):
        if arr.shape != reference_shape:
            msg = f"Shape mismatch: input 0 has shape {reference_shape}, input {i} has shape {arr.shape}"
            raise ValueError(msg)

    # Compute weighted average using dask
    # Convert to float32 for averaging, then back to original dtype
    result = sum(arr.astype(np.float32) * w for arr, w in zip(input_arrays, weights))
    result = da.clip(result, 0, 255).astype(zarr_arrays[0].dtype)

    # Setup output
    out_container, out_dataset = output_path

    # Create output directory structure
    out_ds_path = out_container / out_dataset
    out_ds_path.mkdir(parents=True, exist_ok=True)

    # Copy the source group structure and metadata
    src_group = zarr_groups[0]
    out_group = zarr.open_group(out_ds_path, mode="w")
    if "multiscales" in src_group.attrs:
        out_group.attrs["multiscales"] = src_group.attrs["multiscales"]

    # Document ensemble provenance
    out_group.attrs["ensemble"] = {
        "sources": [str(container / dataset) for container, dataset in input_paths],
        "weights": weights,
    }

    # Create output zarr array using same path as source (e.g., "0" or "s0")
    out_zarr = out_group.zeros(
        array_path,
        shape=reference_shape,
        chunks=zarr_arrays[0].chunks,
        dtype=zarr_arrays[0].dtype,
    )

    logger.info(f"Created output: {out_ds_path}, shape={reference_shape}")

    # Start dask distributed client for progress dashboard
    client = Client()
    logger.info(f"Dask dashboard: {client.dashboard_link}")

    # Compute and store result
    logger.info("Computing ensemble...")
    da.to_zarr(result, out_zarr)

    client.close()

    # Ensure parent zarr groups exist for neuroglancer compatibility
    for parent in list(out_ds_path.parents)[:-1]:
        if parent.suffix == ".zarr" or any(p.suffix == ".zarr" for p in parent.parents):
            zgroup_file = parent / ".zgroup"
            if not zgroup_file.exists():
                parent.mkdir(parents=True, exist_ok=True)
                zgroup_file.write_text('{"zarr_format":2}')

    logger.info(f"Ensemble complete: {out_ds_path}")


@click.group()
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
def cli(log_levels: tuple[str, ...]):
    """Postprocessing tools for predictions."""
    setup_package_logger(log_levels)


@cli.command()
@click.option(
    "-i",
    "--input",
    "inputs",
    multiple=True,
    required=True,
    help="Input prediction path as container:dataset (can specify multiple times)",
)
@click.option(
    "-o",
    "--output",
    required=True,
    help="Output path as container:dataset",
)
@click.option(
    "-w",
    "--weight",
    "weights",
    multiple=True,
    type=float,
    help="Weight for each input (optional, must match number of inputs if specified)",
)
def ensemble(inputs: tuple[str, ...], output: str, weights: tuple[float, ...]):
    """Ensemble multiple predictions by averaging.

    Example usage:

        # Average two predictions with equal weights
        postprocess ensemble -i pred1.zarr:dataset -i pred2.zarr:dataset -o output.zarr:ensemble

        # Weighted average (2:1 ratio)
        postprocess ensemble -i pred1.zarr:dataset -i pred2.zarr:dataset -o output.zarr:ensemble -w 2 -w 1

        # Average predictions from different runs
        postprocess ensemble \\
            -i /path/to/run1/predictions.zarr:it50000/mito \\
            -i /path/to/run2/predictions.zarr:it50000/mito \\
            -i /path/to/run3/predictions.zarr:it50000/mito \\
            -o /path/to/ensemble.zarr:mito
    """
    # Parse input paths
    input_paths = []
    for inp in inputs:
        if ":" not in inp:
            msg = f"Invalid input format '{inp}'. Expected 'container:dataset'"
            raise click.ClickException(msg)
        container, dataset = inp.rsplit(":", 1)
        container_path = Path(container)
        if not container_path.exists():
            msg = f"Input container does not exist: {container}"
            raise click.ClickException(msg)
        input_paths.append((container_path, dataset))

    # Parse output path
    if ":" not in output:
        msg = f"Invalid output format '{output}'. Expected 'container:dataset'"
        raise click.ClickException(msg)
    out_container, out_dataset = output.rsplit(":", 1)

    # Parse weights
    weight_list = list(weights) if weights else None

    ensemble_predictions(
        input_paths,
        (Path(out_container), out_dataset),
        weights=weight_list,
    )


if __name__ == "__main__":
    cli()
