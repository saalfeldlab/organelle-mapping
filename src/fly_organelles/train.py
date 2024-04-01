import gunpowder as gp
from fly_organelles.data import CellMapCropSource, ExtractMask
from fly_organelles.model import MaskedMultiLabelBCEwithLogits
import torch
import corditea
import math


def make_data_pipeline(
    labels: list[str],
    label_stores: list[str],
    raw_stores: list[str],
    pad_width: gp.Coordinate,
    sampling: tuple[int],
    batch_size: int = 5,
):
    raw = gp.ArrayKey("RAW")
    label_keys = {}
    for label in labels:
        label_keys[label] = gp.ArrayKey(label.upper())
    srcs = []
    probs = []
    for label_store, raw_store in zip(label_stores, raw_stores):
        src = CellMapCropSource(label_store, raw_store, label_keys, raw, sampling)
        probs.append(src.get_size())
        for label_key in label_keys.values():
            print(label_key)
            src += gp.Pad(label_key, pad_width, value=255)
        src += gp.Normalize(raw)

        # TODO CONTRAST ADJUSTMENT
        srcs.append(src)
    pipeline = tuple(srcs) + gp.RandomProvider(probs) + gp.RandomLocation()
    pipeline += corditea.GaussianNoiseAugment(raw, noise_prob=0.75)
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
    label_stores: list[str],
    raw_stores: list[str],
    pad_width: gp.Coordinate,
    sampling: tuple[int],
    batch_size: int = 5,
):
    pipeline = make_data_pipeline(
        labels,
        label_stores,
        raw_stores,
        pad_width,
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
