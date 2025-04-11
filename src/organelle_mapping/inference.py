import logging
import os
import subprocess

import click
import daisy
import numpy as np
import torch
from funlib.persistence import Array, open_ds, prepare_ds
from skimage.transform import rescale

from organelle_mapping.model import load_eval_model

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def cli(log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))


def spawn_worker(
    checkpoint,
    num_outputs,
    channels,
    voxel_size,
    out_container,
    out_dataset,
    in_container,
    in_dataset,
    billing,
    local=True,
    min_raw=0,
    max_raw=255,
    mask_containers=list(),
    mask_datasets=list(),
    instance: bool = False,
    time_limit: int = 2000,
):
    def run_worker():
        mask_args = []
        for mask_container, mask_dataset in zip(mask_containers, mask_datasets):
            mask_args.extend(["-mc", mask_container, "-md", mask_dataset])
        process_cmd = [
            "inference",
            "start-worker",
            "-ckpt",
            f"{checkpoint}",
            "-no",
            f"{num_outputs}",
            "-cs",
            f"{channels}",
            "-vs",
            *map(str, voxel_size),
            "-oc",
            f"{out_container}",
            "-od",
            f"{out_dataset}",
            "-ic",
            f"{in_container}",
            "-id",
            f"{in_dataset}",
            "--min-raw",
            f"{min_raw}",
            "--max-raw",
            f"{max_raw}",
            "--instance",
            f"{instance}",
            *mask_args,
        ]
        if local:
            subprocess.run(process_cmd)
        else:
            subprocess.run(
                [
                    "bsub",
                    "-P",
                    f"{billing}",
                    "-J",
                    "pred",
                    "-q",
                    "gpu_rtx8000",
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
                ]
            )

    return run_worker


@cli.command()
@click.option("-ckpt", "--checkpoint", type=str)
@click.option("-no", "--num_outputs", type=int)
@click.option(
    "-cs", "--channels", type=str
)  # 0:ecs,1:plasma_membrane,2:ecs (comma seperated number:dataset_name)
@click.option("-is", "--input_shape", nargs=3, type=int)
@click.option("-os", "--output_shape", nargs=3, type=int)
@click.option("-vs", "--voxel_size", nargs=3, type=int)
@click.option("-oc", "--out_container", type=click.Path(file_okay=False))  # zarr file
@click.option("-od", "--out_dataset", type=str)  # zarr group
@click.option("-ic", "--in_container", type=click.Path(exists=True, file_okay=False))
@click.option("-id", "--in_dataset", type=str)
@click.option("-w", "--workers", type=int, default=1)  # number of workers
@click.option(
    "-roi",
    "--roi",
    type=str,
    required=False,
    help="The roi to predict on. Passed in as [lower:upper, lower:upper, ... ]",
)  # "[0:480000,10000:120000,15000:180000]" (in world coordinates)
@click.option("--local/--bsub", default=True)
@click.option("--billing", default=None)
@click.option("--min-raw", type=float, default=0)
@click.option("--max-raw", type=float, default=255)
@click.option(
    "-mc",
    "--mask-container",
    type=click.Path(file_okay=False),
    multiple=True,
    default=list(),
)  # ignore
@click.option(
    "-md",
    "--mask-dataset",
    type=click.Path(file_okay=False),
    multiple=True,
    default=list(),
)  # ignore
@click.option("--instance", type=bool, default=False)  # this is for affinities
@click.option(
    "-tw", "--per-block-time-estimate", type=float, default=30, help="in seconds"
)
def predict(
    checkpoint,
    num_outputs,
    channels,
    input_shape,
    output_shape,
    voxel_size,
    out_container,
    out_dataset,
    in_container,
    in_dataset,
    workers,
    roi,
    local,
    billing,
    min_raw,
    max_raw,
    mask_container,
    mask_dataset,
    instance,
    per_block_time_estimate,
):
    if not local:
        assert billing is not None
    parsed_channels = [channel.split(":") for channel in channels.split(",")]
    assert os.path.exists(checkpoint)
    raw = open_ds(os.path.join(in_container, in_dataset), voxel_size=voxel_size)

    if roi is not None:
        parsed_start, parsed_end = zip(
            *[
                tuple(int(coord) for coord in axis.split(":"))
                for axis in roi.strip("[]").split(",")
            ]
        )
        parsed_roi = daisy.Roi(
            daisy.Coordinate(parsed_start),
            daisy.Coordinate(parsed_end) - daisy.Coordinate(parsed_start),
        )
    else:
        parsed_roi = raw.roi  # * daisy.Coordinate(voxel_size)

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
                num_channels = len(
                    range(int(indexes.split("-")[0]), int(indexes.split("-")[1]))
                )
                shape = (num_channels,) + tuple(
                    (total_write_roi / output_voxel_size).shape
                )
                chunk_shape = (num_channels,) + tuple(
                    (write_roi / output_voxel_size).shape
                )
                axes = ["c^"] + ["z", "y", "x"]
            prepare_ds(
                f"{out_container}/{out_dataset}/{channel}",
                shape=shape,
                # offset = total_write_roi.get_offset(),
                voxel_size=output_voxel_size,
                axis_names=axes,
                units=["nm", "nm", "nm"],
                chunk_shape=chunk_shape,
                dtype=np.uint8,
                mode="w",
            )
    else:
        raise NotImplementedError("Instance prediction not implemented yet")
        num_channels = num_outputs
        assert len(parsed_channels) == 1
        indexes, channel = parsed_channels[0]
        for i in range(0, num_channels, 3):
            prepare_ds(
                f"{out_container}/{out_dataset}/{channel}__{i}",
                shape=(num_channels, *(total_write_roi / output_voxel_size).shape),
                offset=total_write_roi.get_offset(),
                voxel_size=output_voxel_size,
                axis_names=["c^", "z", "y", "x"],
                units=["nm", "nm", "nm"],
                chunk_shape=(num_channels, *write_roi.shape),
                dtype=np.float32,
            )

    task = daisy.Task(
        "test_server_task",
        total_roi=total_read_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=spawn_worker(
            checkpoint,
            num_outputs,
            channels,
            voxel_size,
            out_container,
            out_dataset,
            in_container,
            in_dataset,
            billing,
            local,
            min_raw,
            max_raw,
            mask_container,
            mask_dataset,
            instance,
            time_limit,
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
@click.option("-ckpt", "--checkpoint", type=str)
@click.option("-no", "--num_outputs", type=int)
@click.option("-cs", "--channels", type=str)
@click.option("-vs", "--voxel_size", nargs=3, type=int)
@click.option("-oc", "--out_container", type=click.Path(exists=True, file_okay=False))
@click.option("-od", "--out_dataset", type=str)
@click.option("-ic", "--in_container", type=click.Path(exists=True, file_okay=False))
@click.option("-id", "--in_dataset", type=str)
@click.option("--min-raw", type=float, default=0)
@click.option("--max-raw", type=float, default=255)
@click.option(
    "-mc",
    "--mask-container",
    type=click.Path(file_okay=False),
    multiple=True,
    default=list(),
)
@click.option(
    "-md",
    "--mask-dataset",
    type=click.Path(file_okay=False),
    multiple=True,
    default=list(),
)
@click.option(
    "--instance",
    type=bool,
    default=False,
)
def start_worker(
    checkpoint,
    num_outputs,
    channels,
    voxel_size,
    out_container,
    out_dataset,
    in_container,
    in_dataset,
    min_raw,
    max_raw,
    mask_container,
    mask_dataset,
    instance,
):
    shift = min_raw
    scale = max_raw - min_raw
    parsed_channels = [channel.split(":") for channel in channels.split(",")]

    client = daisy.Client()

    model = load_eval_model(num_outputs, checkpoint)
    device = next(model.parameters()).device
    raw_dataset = open_ds(os.path.join(in_container, in_dataset), voxel_size=voxel_size)
    mask_datasets = [
        open_ds(mc, md, voxel_size=voxel_size)
        for mc, md in zip(mask_container, mask_dataset)
    ]

    # voxel_size = raw_dataset.voxel_size
    output_voxel_size = daisy.Coordinate(voxel_size)
    num_channels = num_outputs
    if not instance:
        out_datasets = [
            open_ds(
                f"{out_container}/{out_dataset}/{channel}",
                mode="r+",
                voxel_size=output_voxel_size,
            )
            for _, channel in parsed_channels
        ]
    else:
        raise NotImplementedError("Instance prediction not implemented yet")
        assert len(parsed_channels) == 1
        indexes, channel = parsed_channels[0]
        out_datasets = [
            open_ds(out_container, f"{out_dataset}/{channel}__{i}", mode="r+")
            for i in range(0, num_channels, 3)
        ]

    while True:
        with client.acquire_block() as block:
            if block is None:
                break
            if len(mask_datasets) > 0:
                mask_data = any(
                    [
                        np.any(
                            mask_dataset.to_ndarray(
                                roi=block.read_roi.snap_to_grid(
                                    mask_dataset.voxel_size
                                ),
                                fill_value=0,
                            )
                        )
                        for mask_dataset in mask_datasets
                    ]
                )
            else:
                mask_data = 1
            if not np.any(mask_data):
                # avoid predicting if mask is empty
                continue

            if not instance:
                raw_input = (
                    2.0
                    * (
                        raw_dataset.to_ndarray(
                            roi=block.read_roi, fill_value=shift + scale
                        ).astype(np.float32)
                        - shift
                    )
                    / scale
                ) - 1.0
            else:
                raise NotImplementedError("Instance prediction not implemented yet")
                raw_input = (
                    raw_dataset.to_ndarray(
                        roi=block.read_roi, fill_value=shift + scale
                    ).astype(np.float32)
                    - shift
                ) / scale
            raw_input = np.expand_dims(raw_input, (0, 1))
            write_roi = block.write_roi.intersect(out_datasets[0].roi)

            if out_datasets[0].to_ndarray(write_roi).any():
                # block has already been processed
                continue

            with torch.no_grad():
                predictions = Array(
                    model.forward(torch.from_numpy(raw_input).float().to(device))
                    .detach()
                    .cpu()
                    .numpy()[0],
                    block.write_roi.offset,
                    output_voxel_size,
                    axis_names=["c^", "z", "y", "x"],
                )

                write_data = predictions.to_ndarray(write_roi).clip(0, 1)
                if not instance:
                    write_data = (write_data) * 255.0  # / 2.0
                    for (i, _), out_dataset in zip(parsed_channels, out_datasets):
                        indexes = []
                        if "-" in i:
                            indexes = [int(j) for j in i.split("-")]
                        else:
                            indexes = [int(i)]
                        if len(indexes) > 1:
                            out_dataset[write_roi] = np.stack(
                                [write_data[j] for j in indexes], axis=0
                            ).astype(np.uint8)
                        else:
                            out_dataset[write_roi] = write_data[indexes[0]].astype(
                                np.uint8
                            )
                else:
                    for i, out_dataset in zip(range(0, num_channels, 3), out_datasets):
                        out_dataset[write_roi] = write_data[i : i + 3].astype(
                            np.float32
                        )

            block.status = daisy.BlockStatus.SUCCESS
