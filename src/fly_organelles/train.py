import gunpowder as gp
from fly_organelles.data import CellMapCropSource, ExtractMask
from fly_organelles.model import MaskedMultiLabelBCEwithLogits
import torch
import corditea
import math
import logging
import os
import numpy as np

logger = logging.getLogger("__name__")


def make_data_pipeline(
    labels: list[str],
    datasets: dict,
    pad_width_in: gp.Coordinate,
    pad_width_out: gp.Coordinate,
    sampling: tuple[int],
    batch_size: int = 5
):  
    logger.debug(f"Using pad_with {pad_width_in} for input and {pad_width_out} for output arrays.")
    raw = gp.ArrayKey("RAW")
    label_keys = {}
    for label in labels:
        label_keys[label] = gp.ArrayKey(label.upper())
    srcs = []
    probs = []
    factors = {np.dtype("uint8"): 255, np.dtype("uint16"): 2**16-1}
    for dataset, ds_info in datasets["datasets"].items():
        for crops in ds_info["crops"]:
            for crop in crops.split(","):
                src = CellMapCropSource(
                    os.path.join(datasets["gt_path"], dataset, "groundtruth.zarr", crop),
                    ds_info["raw"],
                    label_keys,
                    raw,
                    sampling
                )
                src_pipe = src
                if src.needs_downsampling:
                    src_pipe += corditea.AverageDownSample(raw,sampling)
                probs.append(src.get_size()/len(crops.split(",")))
                for label_key in label_keys.values():
                    src_pipe += gp.Pad(label_key, pad_width_out, value=255)
                factor = factors[src.specs[raw].dtype]
                src_pipe += gp.Normalize(raw, factor=1./factor)
                minc, maxc = ds_info["contrast"]
                src_pipe += gp.IntensityScaleShift(raw, scale= (maxc-minc)/factor, shift = minc/factor)
                src_pipe += gp.Pad(raw, pad_width_in, value=0)
                src_pipe += gp.RandomLocation()
                srcs.append(src_pipe)
                
    pipeline = tuple(srcs) + gp.RandomProvider(probs)
    pipeline += gp.IntensityAugment(raw, 0.75, 1.5, -0.15, 0.15)
    pipeline += corditea.GammaAugment([raw], 0.75, 4 / 3.0)
    pipeline += gp.SimpleAugment()
    pipeline += corditea.ElasticAugment(
        control_point_spacing=gp.Coordinate((100, 100, 100)),
        control_point_displacement_sigma=gp.Coordinate((12, 12, 12)),
        rotation_interval=(0, math.pi / 2.0),
        subsample=8,
        uniform_3d_rotation=True,
    )
    pipeline += gp.IntensityScaleShift(raw, 2, -1)
    pipeline += gp.Unsqueeze(list(label_keys.values()))
    pipeline += corditea.Concatenate(list(label_keys.values()), gp.ArrayKey("LABELS"))
    pipeline += ExtractMask(gp.ArrayKey("LABELS"), gp.ArrayKey("MASK"))
    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(batch_size)
    pipeline += gp.AsType(gp.ArrayKey("LABELS"), "float32")

    return pipeline


def make_train_pipeline(
    model,
    labels: list[str],
    label_weights: list[float],
    datasets: dict,
    pad_width_in: gp.Coordinate,
    pad_width_out: gp.Coordinate,
    sampling: tuple[int],
    batch_size: int = 5,
):
    pipeline = make_data_pipeline(
        labels,
        datasets,
        pad_width_in,
        pad_width_out,
        sampling,
        batch_size,
    )
    pipeline += gp.torch.Train(
        model=model,
        loss=MaskedMultiLabelBCEwithLogits(label_weights),
        optimizer=torch.optim.Adam(lr=0.5e-4, params=model.parameters()),
        inputs={"raw": gp.ArrayKey("RAW")},
        loss_inputs={"output": gp.ArrayKey("OUTPUT"), "target": gp.ArrayKey("LABELS"), "mask": gp.ArrayKey("MASK")},
        outputs={0: gp.ArrayKey("OUTPUT")},
        device="cuda:1",
    )
    pipeline += gp.Snapshot(
        {
            gp.ArrayKey("LABELS"): "labels",
            gp.ArrayKey("RAW"): "raw",
            gp.ArrayKey("MASK"): "mask",
            gp.ArrayKey("OUTPUT"): "output",
        }
    )
    return pipeline
