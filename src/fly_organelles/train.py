import logging
import math
import os

import corditea
import gunpowder as gp
import numpy as np
import torch

from fly_organelles.data import CellMapCropSource, ExtractMask

from fly_organelles.model import MaskedMultiLabelBCEwithLogits, WeightedMSELoss, AffinitiesLoss

# from lsd.train.gp import AddLocalShapeDescriptor

from fly_organelles.utils import ShiftNorm, Binarize, Distance
from fly_organelles.lsds.gp_node import LSDAffinities

logger = logging.getLogger("__name__")


def sigmoidify(arr):
    return torch.nn.functional.sigmoid(torch.tensor(arr)).numpy()

def make_affinities_data_pipeline(
    labels: list[str],
    datasets: dict,
    pad_width_out: gp.Coordinate,
    sampling: tuple[int],
    max_out_request: gp.Coordinate,
    displacement_sigma: gp.Coordinate,
    batch_size: int = 5,
    min_mask: float = None,
    affinities_map = [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],]
):
    raw = gp.ArrayKey("RAW")
    all_labels_key = gp.ArrayKey("LABELS")
    
    label_keys = {}
    for label in labels:
        label_keys[label] = gp.ArrayKey(label.upper())
    
    
    srcs = []
    probs = []
    factors = {
        np.dtype("uint8"): 255,
        np.dtype("uint16"): 2**16 - 1,
        np.dtype("int16"): 2**15 - 1  # 32767
    }
    for dataset, ds_info in datasets["datasets"].items():
        for crop in ds_info["crops"]:
            # for crop in crops.split(","):
            src = CellMapCropSource(
                ds_info["crops"][crop],
                ds_info["raw"],
                label_keys,
                raw,
                sampling,
                base_padding=pad_width_out,
                max_request=max_out_request,
            )
            src_pipe = src
            if src.needs_downsampling:
                src_pipe += corditea.AverageDownSample(raw, sampling)
            probs.append(src.get_size() / len(ds_info["crops"]))
            # logging.debug(f"Padding {crop} with {src.padding}")

            minc, maxc = ds_info["contrast"]
            src_pipe+= gp.AsType(raw, "float32")
            src_pipe+= ShiftNorm(raw, minc, maxc)
            # src_pipe += gp.IntensityScaleShift(raw, scale=1/(maxc - minc), shift=-(minc/(maxc-minc)) )

            # src_pipe += gp.IntensityScaleShift(raw, scale=(maxc - minc) / factor, shift=minc / factor)
            src_pipe += gp.Pad(raw, None, value=0)
            src_pipe += gp.RandomLocation()

            srcs.append(src_pipe)

    pipeline = tuple(srcs) + gp.RandomProvider(probs)
    
    for label_key in label_keys.values():
        if min_mask is not None:
            pipeline += gp.Reject(label_key,min_masked=min_mask)
        pipeline += LSDAffinities(
            label_key,
            affinities=affinities_map,
            )
    src_pipe += gp.AsType(label_key, "float32")
    src_pipe += gp.Pad(label_key, src.padding, value=255.0)
    pipeline += gp.IntensityAugment(raw, 0.75, 1.5, -0.15, 0.15)
    pipeline += corditea.GammaAugment([raw], 0.75, 4 / 3.0)
    pipeline += gp.SimpleAugment()
    pipeline += corditea.ElasticAugment(
        control_point_spacing=gp.Coordinate((25, 25, 25)),
        control_point_displacement_sigma=displacement_sigma,
        rotation_interval=(0, math.pi / 2.0),
        subsample=8,
        uniform_3d_rotation=True,
        augmentation_probability=0.6,
    )
    pipeline += gp.IntensityScaleShift(raw, 2, -1)
    pipeline += corditea.GaussianNoiseAugment(raw, var_range=(0, 0.01), noise_prob=0.5)
    # pipeline += gp.Unsqueeze(list(affinities_keys.values()))
    pipeline += corditea.Concatenate(list(label_keys.values()), all_labels_key)
    
    pipeline += gp.Pad(all_labels_key, src.padding, value=255)
    # pipeline += gp.BalanceLabels(
    #     all_labels_key,
    #     gp.ArrayKey("MASK"))
    pipeline += ExtractMask(all_labels_key, gp.ArrayKey("MASK"))
    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(batch_size)
    pipeline += gp.AsType(all_labels_key, "float32")

    return pipeline


def make_data_pipeline(
    labels: list[str],
    datasets: dict,
    pad_width_out: gp.Coordinate,
    sampling: tuple[int],
    max_out_request: gp.Coordinate,
    displacement_sigma: gp.Coordinate,
    batch_size: int = 5,
    min_mask: float = None,
    distance_sigma : float = None,
):
    raw = gp.ArrayKey("RAW")
    all_labels_key = gp.ArrayKey("LABELS")
    label_keys = {}
    for label in labels:
        label_keys[label] = gp.ArrayKey(label.upper())
    srcs = []
    probs = []
    factors = {
        np.dtype("uint8"): 255,
        np.dtype("uint16"): 2**16 - 1,
        np.dtype("int16"): 2**15 - 1  # 32767
    }
    for dataset, ds_info in datasets["datasets"].items():
        for crop in ds_info["crops"]:
            src = CellMapCropSource(
                ds_info["crops"][crop],
                ds_info["raw"],
                label_keys,
                raw,
                sampling,
                base_padding=pad_width_out,
                max_request=max_out_request,
            )
            src_pipe = src
            # if src.needs_downsampling:
            #     src_pipe += corditea.AverageDownSample(raw, sampling)
            probs.append(src.get_size() / len(ds_info["crops"]))
            logging.debug(f"Padding {crop} with {src.padding}")
            for label_key in label_keys.values():
                if distance_sigma is None:
                    src_pipe += Binarize(label_key)
                else:
                    src_pipe+= Distance(label_key, sigma=distance_sigma)
                
                src_pipe += gp.AsType(label_key, "float32")
            factor = factors[src.specs[raw].dtype]
            # src_pipe += gp.Normalize(raw, factor=1.0 / factor)
            # src_pipe += gp.Normalize(raw, factor=1.0)
            minc, maxc = ds_info["contrast"]
            src_pipe+= gp.AsType(raw, "float32")
            src_pipe+= ShiftNorm(raw, minc, maxc)
            # src_pipe += gp.IntensityScaleShift(raw, scale=1/(maxc - minc), shift=-(minc/(maxc-minc)) )

            # src_pipe += gp.IntensityScaleShift(raw, scale=(maxc - minc) / factor, shift=minc / factor)
            src_pipe += gp.Pad(raw, None, value=0)
            src_pipe += gp.RandomLocation()
            
            srcs.append(src_pipe)
    
    pipeline = tuple(srcs) + gp.RandomProvider(probs)
    pipeline += gp.Unsqueeze(list(label_keys.values()))
    pipeline += corditea.Concatenate(list(label_keys.values()), all_labels_key)
    if min_mask is not None:
        pipeline += gp.Reject(all_labels_key,min_masked=min_mask)
    pipeline += gp.Pad(all_labels_key, src.padding, value=255)
    pipeline += gp.IntensityAugment(raw, 0.75, 1.5, -0.15, 0.15)
    pipeline += corditea.GammaAugment([raw], 0.75, 4 / 3.0)
    pipeline += gp.SimpleAugment()
    pipeline += corditea.ElasticAugment(
        control_point_spacing=gp.Coordinate((25, 25, 25)),
        control_point_displacement_sigma=displacement_sigma,
        rotation_interval=(0, math.pi / 2.0),
        subsample=8,
        uniform_3d_rotation=True,
        augmentation_probability=0.6,
    )
    pipeline += gp.IntensityScaleShift(raw, 2, -1)
    pipeline += corditea.GaussianNoiseAugment(raw, var_range=(0, 0.01), noise_prob=0.5)

    
    pipeline += ExtractMask(all_labels_key, gp.ArrayKey("MASK"))
    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(batch_size)
    pipeline += gp.AsType(all_labels_key, "float32")

    return pipeline


def make_train_pipeline(
    model,
    labels: list[str],
    label_weights: list[float],
    datasets: dict,
    pad_width_out: gp.Coordinate,
    sampling: tuple[int],
    max_out_request: gp.Coordinate,
    displacement_sigma: gp.Coordinate,
    input_size: gp.Coordinate,
    output_size: gp.Coordinate,
    batch_size: int = 5,
    l_rate: float = 0.5e-4,
    log_dir: str = "logs",
    affinities = False,
    affinities_map = None,
    min_mask = None,
    distance_sigma = None,
):
    if affinities:
        pipeline = make_affinities_data_pipeline(
            labels,
            datasets,
            pad_width_out,
            sampling,
            max_out_request,
            displacement_sigma,
            batch_size,
            affinities_map= affinities_map,
            min_mask=min_mask,
        )
        # loss_fn = BalancedAffinitiesLoss(num_affinities_channels=len(affinities_map))
        loss_fn = AffinitiesLoss(nb_affinities=len(affinities_map))
    else:
        pipeline = make_data_pipeline(
            labels,
            datasets,
            pad_width_out,
            sampling,
            max_out_request,
            displacement_sigma,
            batch_size,
            min_mask=min_mask,
            distance_sigma = distance_sigma,
        )
        loss_fn = WeightedMSELoss(foreground_factor=2)
    pipeline += gp.PreCache(30, 20)

    
    
    pipeline += gp.torch.Train(
    model=model,
    loss=loss_fn,
    optimizer=torch.optim.Adam(lr=l_rate, params=model.parameters()),
    # optimizer = torch.optim.AdamW(model.parameters(), lr=l_rate, weight_decay=1e-5),
    inputs={"raw": gp.ArrayKey("RAW")},
    loss_inputs={"output": gp.ArrayKey("OUTPUT"), "target": gp.ArrayKey("LABELS"), "mask": gp.ArrayKey("MASK")},
    outputs={0: gp.ArrayKey("OUTPUT")},
    device="cuda",
    log_every=20,
    log_dir=log_dir,
    save_every=2000,
)
    
    pipeline += corditea.LambdaFilter(sigmoidify, gp.ArrayKey("OUTPUT"), gp.ArrayKey("NORM_OUTPUT"))
    snapshot_request = gp.BatchRequest()
    snapshot_request.add(gp.ArrayKey("DUMMY"), input_size, voxel_size=gp.Coordinate(sampling))
    snapshot_request.add(gp.ArrayKey("NORM_OUTPUT"), output_size, voxel_size=gp.Coordinate(sampling))
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
