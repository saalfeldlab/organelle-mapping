import logging
import os
import subprocess
from pathlib import Path

import click
import daisy
import numpy as np
import torch
import yaml
import zarr
from funlib.persistence import Array, open_ds
from funlib.persistence import open_ome_ds as funlib_open_ome_ds

from organelle_mapping.config.inference import InferenceConfig
from organelle_mapping.model import load_eval_model
from organelle_mapping.neuroglancer_state import state_from_inference_config
from organelle_mapping.utils import setup_package_logger

logger = logging.getLogger(__name__)


def prepare_ome_ds(
    store: Path,
    name: str,
    shape: tuple,
    offset: tuple | None = None,
    voxel_size: tuple | None = None,
    axis_names: list[str] | None = None,
    units: list[str] | None = None,
    chunk_shape: tuple | None = None,
    dtype=np.float32,
) -> Array:
    """Prepare an OME-Zarr dataset by directly creating zarr with OME metadata.

    This bypasses funlib.persistence and iohub bugs by creating the zarr
    structure directly with proper OME-NGFF 0.4 metadata.
    """
    # Default values
    if offset is None:
        offset = (0,) * len(shape)
    if voxel_size is None:
        voxel_size = (1,) * min(3, len(shape))
    if axis_names is None:
        axis_names = ["z", "y", "x"][-len(shape) :]
    if units is None:
        units = ["nanometer"] * min(3, len(shape))
    if chunk_shape is None:
        chunk_shape = shape

    # Ensure parent zarr groups exist (needed for neuroglancer compatibility)
    # Create .zgroup files at each level of the hierarchy
    ds_path = store / name
    for parent in list(ds_path.parents)[:-1]:  # All parents except filesystem root
        if parent.suffix == ".zarr" or any(p.suffix == ".zarr" for p in parent.parents):
            # This is inside a zarr store, ensure it's a valid group
            zgroup_file = parent / ".zgroup"
            if not zgroup_file.exists():
                parent.mkdir(parents=True, exist_ok=True)
                zgroup_file.write_text('{"zarr_format":2}')

    # Create the zarr group at store/name
    root = zarr.open_group(ds_path, mode="w")

    # Create the data array at "0" (first scale level)
    root.zeros(
        "0",
        shape=shape,
        chunks=chunk_shape,
        dtype=dtype,
    )

    # Build axes metadata (OME-NGFF 0.4 format) and types list for Array
    axes = []
    types = []
    for ax_name in axis_names:
        if ax_name.startswith("c"):
            axes.append({"name": ax_name.rstrip("^"), "type": "channel"})
            types.append("channel")
        elif ax_name in ["t"]:
            axes.append({"name": ax_name, "type": "time"})
            types.append("time")
        else:
            # Get unit for this spatial axis
            spatial_idx = len([a for a in axes if a.get("type") == "space"])
            unit = units[spatial_idx] if spatial_idx < len(units) else "nanometer"
            axes.append({"name": ax_name, "type": "space", "unit": unit})
            types.append("space")

    # Build coordinate transforms
    # Scale: for non-spatial dims use 1, for spatial dims use voxel_size
    scale = []
    translation = []
    spatial_idx = 0
    for ax_name in axis_names:
        if ax_name.startswith("c") or ax_name == "t":
            scale.append(1)
            translation.append(0)
        else:
            scale.append(voxel_size[spatial_idx] if spatial_idx < len(voxel_size) else 1)
            translation.append(offset[spatial_idx] if spatial_idx < len(offset) else 0)
            spatial_idx += 1

    # Write OME-NGFF multiscales metadata
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": axes,
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": scale},
                        {"type": "translation", "translation": translation},
                    ],
                }
            ],
        }
    ]

    # Return as funlib Array for compatibility
    return Array(
        root["0"],
        offset=offset,
        voxel_size=voxel_size,
        axis_names=axis_names,
        types=types,
    )


def open_ome_ds(store: Path, name: str, mode: str = "r") -> Array:
    """Open an OME-Zarr dataset created by prepare_ome_ds."""
    root = zarr.open_group(store, mode=mode)

    # Parse multiscales metadata
    multiscales = root.attrs.get("multiscales", [{}])[0]
    axes = multiscales.get("axes", [])
    datasets = multiscales.get("datasets", [{}])

    # Find the requested dataset
    ds_meta = next((d for d in datasets if d.get("path") == name), datasets[0] if datasets else {})
    transforms = ds_meta.get("coordinateTransformations", [])

    # Extract scale and translation
    scale = [1] * len(axes)
    translation = [0] * len(axes)
    for t in transforms:
        if t.get("type") == "scale":
            scale = t.get("scale", scale)
        elif t.get("type") == "translation":
            translation = t.get("translation", translation)

    # Extract axis names and types from OME metadata
    axis_names = [a.get("name", f"d{i}") for i, a in enumerate(axes)]
    # Map OME-Zarr types to funlib types (channel vs space)
    types = [a.get("type", "space") for a in axes]

    # Extract spatial dimensions only for offset/voxel_size
    spatial_scale = []
    spatial_offset = []
    for i, ax in enumerate(axes):
        if ax.get("type") == "space":
            spatial_scale.append(scale[i])
            spatial_offset.append(translation[i])

    return Array(
        root[name],
        offset=tuple(spatial_offset) if spatial_offset else (0, 0, 0),
        voxel_size=tuple(spatial_scale) if spatial_scale else (1, 1, 1),
        axis_names=axis_names,
        types=types,
    )


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default="INFO",
)
def cli(log_level):
    setup_package_logger(log_level)


def spawn_worker(
    config,
    billing,
    *,
    local: bool = True,
    mask_containers=None,
    mask_datasets=None,
    instance: bool = False,
    time_limit: int = 2000,
    queue: str = "gpu_rtx8000",
):
    if mask_datasets is None:
        mask_datasets = []
    if mask_containers is None:
        mask_containers = []
    def run_worker():
        mask_args = []
        for mask_container, mask_dataset in zip(mask_containers, mask_datasets):
            mask_args.extend(["-mc", mask_container, "-md", mask_dataset])
        process_cmd = [
            "inference",
            "start-worker",
            "-c",
            f"{config}",
            "--instance",
            f"{instance}",
            *mask_args,
        ]
        if local:
            subprocess.run(process_cmd, check=False)
        else:
            subprocess.run(
                [
                    "bsub",
                    "-P",
                    f"{billing}",
                    "-J",
                    "pred",
                    "-q",
                    queue,
                    "-n",
                    "2",
                    "-gpu",
                    "num=1",
                    "-o",
                    "prediction_logs/out",
                    "-e",
                    "prediction_logs/err",
                    "-W",
                    f"{time_limit}",
                    *process_cmd,
                ], check=False
            )

    return run_worker


@cli.command()
@click.option("-c", "--config", type=click.Path(exists=True), help="Path to inference config YAML file")
@click.option("-w", "--workers", type=int, default=1, help="Number of workers for parallel processing")
@click.option("--local/--bsub", default=True, help="Run locally vs on cluster")
@click.option("--billing", default=None, help="Billing account for cluster runs")
@click.option(
    "-mc",
    "--mask-container",
    type=click.Path(file_okay=False),
    multiple=True,
    default=None,
)  # ignore
@click.option(
    "-md",
    "--mask-dataset",
    type=click.Path(file_okay=False),
    multiple=True,
    default=None,
)  # ignore
@click.option("--instance", type=bool, default=False, help="For affinity predictions")
@click.option("-tw", "--per-block-time-estimate", type=float, default=30, help="Time estimate per block in seconds")
@click.option(
    "-q", "--queue", type=str, default="gpu_rtx8000", help="GPU queue for cluster runs (e.g., gpu_rtx8000, gpu_a100)"
)
def predict(
    config,
    workers,
    local,
    billing,
    mask_container,
    mask_dataset,
    instance,
    per_block_time_estimate,
    queue,
):
    if not local and billing is None:
        msg = "billing account is required for cluster runs"
        raise ValueError(msg)
    if mask_container is None:
        mask_container = []
    if mask_dataset is None:
        mask_dataset = []
    # Load config
    inference_config = InferenceConfig.model_validate(
        yaml.safe_load(open(config)), context={"base_dir": Path(config).parent}
    )

    # Extract values from config
    checkpoint = inference_config.checkpoint
    if not os.path.exists(checkpoint):
        msg = f"checkpoint does not exist: {checkpoint}"
        raise FileNotFoundError(msg)

    # Parse channels from output config
    ",".join([f"{out.channels}:{out.name}" for out in inference_config.output_data.outputs])
    parsed_channels = [[out.channels, out.name] for out in inference_config.output_data.outputs]

    # Get architecture details
    input_shape = inference_config.architecture.input_shape
    output_shape = inference_config.architecture.output_shape

    # Get input/output details
    in_container = inference_config.input_data.container
    in_dataset = inference_config.input_data.dataset
    in_scale = inference_config.input_data.scale
    voxel_size = inference_config.input_data.voxel_size

    out_container = inference_config.output_data.container
    out_dataset = inference_config.output_data.dataset

    # Open raw dataset (OME-Zarr if scale specified, otherwise regular zarr)
    if in_scale is not None:
        raw = funlib_open_ome_ds(Path(in_container) / in_dataset, in_scale)
    else:
        raw = open_ds(f"{in_container}/{in_dataset}", voxel_size=voxel_size)

    # Handle ROI
    if inference_config.output_data.roi is not None:
        parsed_roi = daisy.Roi(
            daisy.Coordinate(inference_config.output_data.roi.start),
            daisy.Coordinate(inference_config.output_data.roi.end),
        )
    else:
        parsed_roi = raw.roi

    total_write_roi = raw.roi
    output_voxel_size = daisy.Coordinate(voxel_size)
    read_shape = daisy.Coordinate(input_shape) * raw.voxel_size
    write_shape = daisy.Coordinate(output_shape) * output_voxel_size
    context = (read_shape - write_shape) / 2
    read_roi = daisy.Roi((0,) * read_shape.dims, read_shape)
    write_roi = read_roi.grow(-context, -context)

    total_write_roi = parsed_roi.snap_to_grid(raw.voxel_size)
    total_read_roi = total_write_roi.grow(context, context)
    n_blocks = np.prod((total_write_roi / write_shape).shape)
    time_per_worker = (per_block_time_estimate * n_blocks) / workers
    time_limit = int(np.ceil(time_per_worker / 60.0))
    if not instance:
        for indexes, channel in parsed_channels:
            if "-" not in indexes:
                shape = tuple((total_write_roi / output_voxel_size).shape)
                chunk_shape = tuple((write_roi / output_voxel_size).shape)
                axes = ["z", "y", "x"]
            else:
                # Parse range like "0-9" -> 10 channels (inclusive end)
                start, end = int(indexes.split("-")[0]), int(indexes.split("-")[1])
                num_channels = end - start + 1
                shape = (num_channels, *tuple((total_write_roi / output_voxel_size).shape))
                chunk_shape = (num_channels, *tuple((write_roi / output_voxel_size).shape))
                axes = ["c^", "z", "y", "x"]
            prepare_ome_ds(
                Path(out_container),
                f"{out_dataset}/{channel}",
                shape=shape,
                offset=total_write_roi.get_offset(),
                voxel_size=output_voxel_size,
                axis_names=axes,
                units=["nanometer", "nanometer", "nanometer"],
                chunk_shape=chunk_shape,
                dtype=np.uint8,
            )
    else:
        msg = "Instance prediction not implemented yet"
        raise NotImplementedError(msg)

    task = daisy.Task(
        "test_server_task",
        total_roi=total_read_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=spawn_worker(
            config,
            billing,
            local=local,
            mask_containers=mask_container,
            mask_datasets=mask_dataset,
            instance=instance,
            time_limit=time_limit,
            queue=queue,
        ),
        check_function=None,
        read_write_conflict=False,
        fit="overhang",
        num_workers=workers,
        max_retries=0,
        timeout=None,
    )

    daisy.run_blockwise([task])


@cli.command()
@click.option("-c", "--config", type=click.Path(exists=True), help="Path to inference config YAML file")
@click.option(
    "-mc",
    "--mask-container",
    type=click.Path(file_okay=False),
    multiple=True,
    default=None,
)
@click.option(
    "-md",
    "--mask-dataset",
    type=click.Path(file_okay=False),
    multiple=True,
    default=None,
)
@click.option(
    "--instance",
    type=bool,
    default=False,
)
def start_worker(
    config,
    mask_container,
    mask_dataset,
    instance,
):
    # Load config
    inference_config = InferenceConfig.model_validate(
        yaml.safe_load(open(config)), context={"base_dir": Path(config).parent}
    )
    if mask_container is None:
        mask_container = []
    if mask_dataset is None:
        mask_dataset = []

    # Extract values from config
    checkpoint = inference_config.checkpoint
    in_container = inference_config.input_data.container
    in_dataset = inference_config.input_data.dataset
    in_scale = inference_config.input_data.scale
    voxel_size = inference_config.input_data.voxel_size
    min_raw = inference_config.input_data.min_raw
    max_raw = inference_config.input_data.max_raw
    out_container = inference_config.output_data.container
    out_dataset = inference_config.output_data.dataset

    shift = min_raw
    normalization_scale = max_raw - min_raw
    parsed_channels = [[out.channels, out.name] for out in inference_config.output_data.outputs]

    client = daisy.Client()

    model = load_eval_model(inference_config.architecture, inference_config.targets, checkpoint)
    device = next(model.parameters()).device

    # Open raw dataset (OME-Zarr if scale specified, otherwise regular zarr)
    if in_scale is not None:
        raw_dataset = funlib_open_ome_ds(Path(in_container) / in_dataset, in_scale)
    else:
        raw_dataset = open_ds(f"{in_container}/{in_dataset}", voxel_size=voxel_size)
    mask_datasets = [open_ds(f"{mc}/{md}", voxel_size=voxel_size) for mc, md in zip(mask_container, mask_dataset)]

    # voxel_size = raw_dataset.voxel_size
    output_voxel_size = daisy.Coordinate(voxel_size)
    if not instance:
        out_datasets = [
            open_ome_ds(
                Path(out_container) / out_dataset / channel,
                "0",
                mode="r+",
            )
            for _, channel in parsed_channels
        ]
    else:
        msg = "Instance prediction not implemented yet"
        raise NotImplementedError(msg)

    while True:
        with client.acquire_block() as block:
            if block is None:
                break
            if len(mask_datasets) > 0:
                mask_data = any(
                    np.any(
                            mask_dataset.to_ndarray(
                                roi=block.read_roi.snap_to_grid(mask_dataset.voxel_size),
                                fill_value=0,
                            )
                        )
                        for mask_dataset in mask_datasets
                )
            else:
                mask_data = 1
            if not np.any(mask_data):
                # avoid predicting if mask is empty
                continue

            # Instance case already raised NotImplementedError earlier
            raw_input = (
                2.0
                * (
                    raw_dataset.to_ndarray(roi=block.read_roi, fill_value=shift + normalization_scale).astype(
                        np.float32
                    )
                    - shift
                )
                / normalization_scale
            ) - 1.0
            raw_input = np.expand_dims(raw_input, (0, 1))
            write_roi = block.write_roi.intersect(out_datasets[0].roi)

            if out_datasets[0].to_ndarray(write_roi).any():
                # block has already been processed
                continue

            with torch.no_grad():
                pred_data = model.forward(torch.from_numpy(raw_input).float().to(device)).detach().cpu().numpy()[0]
                predictions = Array(
                    pred_data,
                    block.write_roi.offset,
                    output_voxel_size,
                    axis_names=["c^", "z", "y", "x"],
                    types=["channel", "space", "space", "space"],
                )

                write_data = predictions.to_ndarray(write_roi).clip(0, 1)
                # Instance case already raised NotImplementedError earlier
                write_data = (write_data) * 255.0  # / 2.0
                for (i, _), out_dataset in zip(parsed_channels, out_datasets):
                    if "-" in i:
                        # Parse range like "0-9" -> [0, 1, 2, ..., 9]
                        start, end = [int(j) for j in i.split("-")]
                        indexes = list(range(start, end + 1))
                    else:
                        indexes = [int(i)]
                    if len(indexes) > 1:
                        out_dataset[write_roi] = np.stack([write_data[j] for j in indexes], axis=0).astype(np.uint8)
                    else:
                        out_dataset[write_roi] = write_data[indexes[0]].astype(np.uint8)

            block.status = daisy.BlockStatus.SUCCESS


@cli.command()
@click.option("-c", "--config", type=click.Path(exists=True), help="Path to inference config YAML file")
@click.option(
    "--raw-url",
    type=str,
    default=None,
    help="Fileglancer URL for raw data. The path after TOKEN is matched against the config's container path.",
)
@click.option(
    "--pred-url",
    type=str,
    default=None,
    help="Fileglancer URL for predictions. The path after TOKEN is matched against the config's container path.",
)
@click.option("-o", "--output", type=click.Path(), default=None, help="Output file for JSON state (default: stdout)")
def neuroglancer_state(config, raw_url, pred_url, output):
    """Generate a Neuroglancer JSON state for viewing predictions.

    Reads the inference config to determine raw data location and prediction
    outputs, then generates a JSON state that can be loaded into Neuroglancer.

    The fileglancer URLs should point to any directory in the data path. The
    path after the token (e.g., /fc/files/TOKEN/jrc_fly-mb-1a.zarr) is matched
    against the config's container path to build the final URLs.

    Examples:
        # Generate state for local files
        inference neuroglancer-state -c inference.yaml

        # Generate state with fileglancer URLs (zarr container level)
        inference neuroglancer-state -c inference.yaml \\
            --raw-url https://fileglancer.../fc/files/TOKEN/jrc_fly-mb-1a.zarr \\
            --pred-url https://fileglancer.../fc/files/TOKEN/jrc_fly-mb-1a.zarr

        # Save to file
        inference neuroglancer-state -c inference.yaml -o state.json
    """
    # Generate state
    state = state_from_inference_config(
        config,
        raw_fileglancer_url=raw_url,
        prediction_fileglancer_url=pred_url,
    )

    # Output
    json_output = state.to_json(indent=2)
    if output:
        with open(output, "w") as f:
            f.write(json_output)
        logger.info(f"Wrote Neuroglancer state to {output}")
    else:
        click.echo(json_output)
