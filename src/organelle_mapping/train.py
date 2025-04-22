import logging
import math
import os

import corditea
import gunpowder as gp
import numpy as np
import torch

import organelle_mapping.utils as utils
from organelle_mapping.config import RunConfig
from organelle_mapping.data import CellMapCropSource, ExtractMask
from organelle_mapping.model import MaskedMultiLabelBCEwithLogits

logger = logging.getLogger(__name__)


def sigmoidify(arr):
    return torch.nn.functional.sigmoid(torch.tensor(arr)).numpy()


def make_data_pipeline(
    run: RunConfig,
    input_size: gp.Coordinate,
    output_size: gp.Coordinate,
):
    raw = gp.ArrayKey("RAW")
    label_keys = {}
    for label in run.labels:
        label_keys[label] = gp.ArrayKey(label.upper())
    srcs = []
    probs = []
    factors = {np.dtype("uint8"): 255, np.dtype("uint16"): 2**16 - 1}
    max_out_request = output_size
    for aug in run.augmentations.augmentations:
        if aug.name == "corditea_elastic_augment":
            if aug.rotation_interval[1] > 0:
                logger.info("Adapting maximum output request to account for rotation.")
                max_out_request = gp.Coordinate((np.ceil(np.sqrt(sum(output_size**2))),) * len(output_size))
            if any(cpds > 0 for cpds in aug.control_point_displacement_sigma):
                logger.info("Adapting maximum output request to account for control point displacement.")
                max_out_request += aug.control_point_displacement_sigma * 6
            break

    for _, ds_info in run.data.datasets.items():
        for crops in ds_info.labels.crops:
            for crop in crops.split(","):
                src = CellMapCropSource(
                    os.path.join(ds_info.labels.data, ds_info.labels.group, crop),
                    os.path.join(ds_info.em.data, ds_info.em.group),
                    label_keys,
                    raw,
                    run.sampling,
                    base_padding=output_size / 2.0,
                    max_request=max_out_request,
                )
                src_pipe = src
                if src.needs_downsampling:
                    src_pipe += corditea.AverageDownSample(raw, utils.ax_dict_to_list(run.sampling, src.axes_order))
                probs.append(src.get_size() / len(crops.split(",")))
                logging.debug(f"Padding {crop} with {src.padding}")
                for label_key in label_keys.values():
                    src_pipe += gp.Pad(label_key, src.padding, value=255)
                    src_pipe += gp.AsType(label_key, "float32")
                factor = factors[src.specs[raw].dtype]
                src_pipe += gp.Normalize(raw, factor=1.0 / factor)
                minc, maxc = ds_info.em.contrast
                src_pipe += gp.IntensityScaleShift(raw, scale=(maxc - minc) / factor, shift=minc / factor)
                src_pipe += gp.Pad(raw, None, value=0)
                src_pipe += gp.RandomLocation()
                srcs.append(src_pipe)

    pipeline = tuple(srcs) + gp.RandomProvider(probs)
    for aug in run.augmentations.augmentations:
        pipeline += aug.instantiate()
    pipeline += gp.Unsqueeze(list(label_keys.values()))
    pipeline += corditea.Concatenate(list(label_keys.values()), gp.ArrayKey("LABELS"))
    pipeline += ExtractMask(gp.ArrayKey("LABELS"), gp.ArrayKey("MASK"))
    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(run.batch_size)
    pipeline += gp.AsType(gp.ArrayKey("LABELS"), "float32")

    return pipeline


def make_train_pipeline(run, input_size, output_size):
    pipeline = make_data_pipeline(run, input_size, output_size)
    model = run.architecture.instantiate()
    if run.precache_size > 0:
        pipeline += gp.PreCache(run.precache_size, run.precache_workers)
    pipeline += gp.torch.Train(
        model=model,
        loss=MaskedMultiLabelBCEwithLogits(run.label_weights),
        optimizer=torch.optim.Adam(lr=run.lr, params=model.parameters()),
        inputs={0: gp.ArrayKey("RAW")},
        loss_inputs={
            "output": gp.ArrayKey("OUTPUT"),
            "target": gp.ArrayKey("LABELS"),
            "mask": gp.ArrayKey("MASK"),
        },
        outputs={0: gp.ArrayKey("OUTPUT")},
        device="cuda",
        log_every=run.log_frequency,
        log_dir="logs",
        save_every=run.checkpoint_frequency,
    )
    pipeline += corditea.LambdaFilter(sigmoidify, gp.ArrayKey("OUTPUT"), gp.ArrayKey("NORM_OUTPUT"))
    snapshot_request = gp.BatchRequest()
    snapshot_request.add(
        gp.ArrayKey("DUMMY"),
        input_size,
        voxel_size=gp.Coordinate(list(run.sampling.values())),
    )
    snapshot_request.add(
        gp.ArrayKey("NORM_OUTPUT"),
        output_size,
        voxel_size=gp.Coordinate(list(run.sampling.values())),
    )
    del snapshot_request[gp.ArrayKey("DUMMY")]
    pipeline += gp.Snapshot(
        {
            gp.ArrayKey("LABELS"): "labels",
            gp.ArrayKey("RAW"): "raw",
            gp.ArrayKey("MASK"): "mask",
            gp.ArrayKey("OUTPUT"): "output",
            gp.ArrayKey("NORM_OUTPUT"): "norm_output",
        },
        output_filename="{iteration:08d}.zarr",
        every=500,
        additional_request=snapshot_request,
    )
    return pipeline
