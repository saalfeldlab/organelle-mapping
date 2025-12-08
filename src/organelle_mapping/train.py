import logging
import os

import corditea
import gunpowder as gp
import numpy as np
import torch

from organelle_mapping import utils
from organelle_mapping.config import RunConfig
from organelle_mapping.data import CellMapCropSource, ExtractMask, CollapseAny
from organelle_mapping.loss import CombinedLoss
from organelle_mapping.model import create_sac_model

logger = logging.getLogger(__name__)


def make_data_pipeline(
    run: RunConfig,
    input_size: gp.Coordinate,  # noqa: ARG001
    output_size: gp.Coordinate,
):
    raw = gp.ArrayKey("RAW")
    label_keys = {}
    # Collect all unique source labels from all targets
    all_sources = set()
    for target in run.targets:
        for transform in target.transforms:
            all_sources.add(transform.source)
    for source in all_sources:
        label_keys[source] = gp.ArrayKey(source.upper())
    srcs = []
    probs = []
    factors = {np.dtype("uint8"): 255, np.dtype("uint16"): 2**16 - 1}
    max_out_extent = output_size

    # Account for transform context
    for target in run.targets:
        for transform in target.transforms:
            max_out_extent = transform.adjust_max_extent(max_out_extent)

    # Account for augmentations (in reverse pipeline order)
    for aug in reversed(run.augmentations.augmentations):
        max_out_extent = aug.adjust_max_extent(max_out_extent)

    logger.info(f"Maximum output extent after accounting for transforms and augmentations: {max_out_extent}")

    for ds_name, ds_info in run.data.datasets.items():
        for crops in ds_info.labels.crops:
            for crop in crops.split(","):
                src = CellMapCropSource(
                    os.path.join(ds_info.labels.data, ds_info.labels.group, crop),
                    os.path.join(ds_info.em.data, ds_info.em.group),
                    label_keys,
                    raw,
                    run.sampling,
                    base_padding=output_size / 2.0,
                    max_extent=max_out_extent,
                )
                src_pipe = src
                if src.needs_downsampling:
                    src_pipe += corditea.AverageDownSample(raw, utils.ax_dict_to_list(run.sampling, src.axes_order))
                probs.append(src.get_size() / len(crops.split(",")))
                for label_key in label_keys.values():
                    src_pipe += corditea.Pad(label_key, src.padding, value=255)
                    src_pipe += gp.AsType(label_key, "float32")
                factor = factors[src.specs[raw].dtype]
                src_pipe += gp.Normalize(raw, factor=1.0 / factor)
                minc, maxc = ds_info.em.contrast
                # Map [minc, maxc] to [0, 1] so that after augmentation (*2-1) it becomes [-1, 1]
                src_pipe += gp.IntensityScaleShift(raw, scale=factor / (maxc - minc), shift=-minc / (maxc - minc))
                src_pipe += corditea.Pad(raw, None, value=0)
                src_pipe += gp.RandomLocation()
                srcs.append(src_pipe)

    pipeline = tuple(srcs) + gp.RandomProvider(probs)
    for aug in run.augmentations.augmentations:
        pipeline += aug.instantiate()
    pipeline += gp.Unsqueeze(list(label_keys.values()))

    # Create individual masks for each source label
    label_mask_keys = {}
    for source, label_key in label_keys.items():
        mask_key = gp.ArrayKey(f"{source.upper()}_MASK")
        label_mask_keys[source] = mask_key
        pipeline += ExtractMask(label_key, mask_key)
    # Apply transforms
    for target in run.targets:
        for transform in target.transforms:
            source_key = label_keys[transform.source]
            source_mask_key = label_mask_keys[transform.source]

            transform_nodes = transform.instantiate(source_key, source_mask_key)
            if transform_nodes is not None:
                for tn in transform_nodes:
                    pipeline += tn

    # Ensure consistent dtypes before concatenation
    all_output_keys = sum((target.output_keys for target in run.targets), ())
    for key in all_output_keys:
        pipeline += gp.AsType(key, "float32")
    pipeline += corditea.Concatenate(all_output_keys, gp.ArrayKey("TARGETS"))

    # Create combined MASK array from all target mask keys
    all_mask_keys = sum((target.mask_keys for target in run.targets), ())
    for key in all_mask_keys:
        pipeline += gp.AsType(key, "float32")
    pipeline += corditea.Concatenate(all_mask_keys, gp.ArrayKey("MASK"))
    if run.min_valid_fraction > 0:
        # Create collapsed mask for rejection (valid where ANY label is valid)
        pipeline += CollapseAny(gp.ArrayKey("MASK"), gp.ArrayKey("VALID"))
        pipeline += gp.Reject(
            mask=gp.ArrayKey("VALID"),
            min_masked=run.min_valid_fraction
        )
    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(run.batch_size)
    pipeline += gp.AsType(gp.ArrayKey("TARGETS"), "float32")

    return pipeline


def make_train_pipeline(run, input_size, output_size):
    pipeline = make_data_pipeline(run, input_size, output_size)
    model = create_sac_model(run.architecture, run.targets)
    if run.precache_size > 0:
        pipeline += gp.PreCache(run.precache_size, run.precache_workers)
    pipeline += gp.torch.Train(
        model=model,
        loss=CombinedLoss(run.targets),
        optimizer=torch.optim.Adam(lr=run.lr, params=model.parameters()),
        inputs={0: gp.ArrayKey("RAW")},
        loss_inputs={
            "output": gp.ArrayKey("OUTPUT"),
            "target": gp.ArrayKey("TARGETS"),
            "mask": gp.ArrayKey("MASK"),
        },
        outputs={0: gp.ArrayKey("OUTPUT")},
        device="cuda",
        log_every=run.log_frequency,
        log_dir="logs",
        save_every=run.checkpoint_frequency,
    )
    pipeline += corditea.LogBatch(mask_key=gp.ArrayKey("MASK"), log_every=run.log_frequency, logger=logger)

    # Build channel activations for normalization
    channel_activations = []
    for target in run.targets:
        for transform in target.transforms:
            activation = transform.inference_activation if transform.activation == "Identity" else None
            channel_activations.extend([activation] * transform.num_channels)

    # Normalize output in-place for visualization
    pipeline += corditea.NormalizeOutput(
        gp.ArrayKey("OUTPUT"),
        channel_activations
    )

    # Remove batch dimension for snapshots (extract first sample: BCDHW -> CDHW)
    pipeline += corditea.Unstack(
        arrays=[
            gp.ArrayKey("TARGETS"),
            gp.ArrayKey("RAW"),
            gp.ArrayKey("MASK"),
            gp.ArrayKey("OUTPUT"),
        ],
        index=0,
    )

    pipeline += gp.Snapshot(
        {
            gp.ArrayKey("TARGETS"): "targets",
            gp.ArrayKey("RAW"): "raw",
            gp.ArrayKey("MASK"): "mask",
            gp.ArrayKey("OUTPUT"): "output",
        },
        output_filename="{iteration:08d}.zarr",
        every=run.snapshot_frequency,
    )
    return pipeline
