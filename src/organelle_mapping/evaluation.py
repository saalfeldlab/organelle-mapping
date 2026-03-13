import logging
from pathlib import Path
from typing import Dict, Optional

import click
import fibsem_tools as fst
import numpy as np
import torch
import yaml
import zarr
from funlib.geometry.coordinate import Coordinate
from pydantic import TypeAdapter
from skimage.measure import block_reduce

from organelle_mapping.config.evaluation import EvaluationConfig
from organelle_mapping.database import init_database, insert_result
from organelle_mapping.metrics import dice_score, jaccard
from organelle_mapping.model import load_eval_model
from organelle_mapping.query import query as query_group
from organelle_mapping.utils import find_target_scale

logger = logging.getLogger("organelle_mapping.evaluation")


def normalize_raw(raw: np.ndarray, min_val: float = 0, max_val: float = 255) -> np.ndarray:
    """Normalize raw data to match training normalization.

    Replicates the training pipeline:
      1. gp.Normalize: raw / dtype_factor → [0, 1]
      2. gp.IntensityScaleShift: maps [min_val, max_val] → [0, 1]
      3. IntensityScaleShift augment: scale=2, shift=-1 → [-1, 1]

    Net effect: (raw - min_val) / (max_val - min_val) * 2 - 1
    """
    return (raw.astype(np.float32) - min_val) / (max_val - min_val) * 2 - 1


def predict_block(
    model: torch.nn.Module,
    raw_block: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Predict on a single block of raw data."""
    # Add batch and channel dimensions
    raw_input = np.expand_dims(raw_block, (0, 1))

    with torch.no_grad():
        predictions = model(torch.from_numpy(raw_input).float().to(device))
        predictions = predictions.detach().cpu().numpy()[0]

    return predictions


def predict_crop(
    model: torch.nn.Module,
    device: torch.device,
    network_config,
    raw_xarray,
    start_coord,
    end_coord,
    voxel_size_coord,
    min_raw: float,
    max_raw: float,
    num_channels: int,
) -> np.ndarray:
    """Predict on an entire crop using blockwise processing."""
    logger.info(f"Start coord {start_coord}, end coord {end_coord}")
    block_input_shape = Coordinate(network_config.input_shape)
    block_output_shape = Coordinate(network_config.output_shape)
    block_input_shape_world = block_input_shape * voxel_size_coord
    block_output_shape_world = block_output_shape * voxel_size_coord

    context = (block_input_shape_world - block_output_shape_world) / 2.0
    raw_begin = Coordinate(raw_xarray[0, 0, 0].coords[ax].item() for ax in raw_xarray.dims)
    raw_end = Coordinate(raw_xarray[-1, -1, -1].coords[ax].item() for ax in raw_xarray.dims)
    raw_resolution = Coordinate(raw_xarray[1, 1, 1].coords[ax].item() for ax in raw_xarray.dims) - raw_begin
    shape = (end_coord - start_coord + raw_resolution) / voxel_size_coord
    logger.info(f"Calculated shape as {shape}")
    # Initialize prediction array for all model output channels
    predictions = np.zeros((num_channels, *tuple(shape)), dtype=np.float32)

    # Process in blocks (end_coord is inclusive, so add one voxel to include it in the range)
    for z_start in range(start_coord[0], end_coord[0] + 1, block_output_shape_world[0]):
        for y_start in range(start_coord[1], end_coord[1] + 1, block_output_shape_world[1]):
            for x_start in range(start_coord[2], end_coord[2] + 1, block_output_shape_world[2]):
                # Calculate world-space ROI for this block
                block_begin_world = Coordinate((z_start, y_start, x_start))
                # Calculate world-space input ROI for this block
                input_begin_world = block_begin_world - context
                input_end_world = input_begin_world + block_input_shape_world - raw_resolution
                begin_oob = [s - r for s, r in zip(input_begin_world, raw_begin)]
                end_oob = [r - s for s, r in zip(input_end_world, raw_end)]

                # Create slices for xarray
                slices = tuple(slice(s, e) for s, e in zip(input_begin_world, input_end_world))
                # Read from xarray
                block_raw = raw_xarray.loc[slices].values
                logger.debug(f"Read raw data of shape {block_raw.shape}")
                if raw_resolution != voxel_size_coord:
                    downsample_factor = tuple(int(voxel_size_coord[i] / raw_resolution[i]) for i in range(3))
                    block_raw = block_reduce(block_raw, downsample_factor, func=np.mean)

                # Check if padding is needed due to boundary issues
                if any(oob < 0 for oob in begin_oob) or any(oob < 0 for oob in end_oob):
                    # Calculate padding needed in voxels for each dimension
                    pad_width = []
                    for i in range(3):
                        # Padding before: convert negative begin_oob to voxels
                        pad_before = max(0, int(-begin_oob[i] / voxel_size_coord[i]))
                        # Padding after: convert negative end_oob to voxels
                        pad_after = max(0, int(-end_oob[i] / voxel_size_coord[i]))
                        pad_width.append((pad_before, pad_after))

                    # Pad the block to expected input shape
                    block_raw = np.pad(block_raw, pad_width, mode="constant", constant_values=min_raw)
                    logger.debug(f"Padded block from {block_raw.shape} with pad_width {pad_width}")

                logger.debug(f"Block shape {block_raw.shape}")
                # Handle fill value for out-of-bounds areas
                # if block_raw.size == 0:
                #     block_raw = np.full(tuple(shape // voxel_size_coord), min_raw)
                block_raw_norm = normalize_raw(block_raw, min_raw, max_raw)

                logger.debug(f"Block raw shape after read: {block_raw_norm.shape}")

                # Predict this block
                block_predictions = predict_block(model, block_raw_norm, device)

                # Place predictions in output array (crop if needed)
                pred_start_vx = (block_begin_world - start_coord) / voxel_size_coord
                pred_end_vx = pred_start_vx + block_output_shape
                pred_end_vx = [min(pred_end_vx[i], shape[i]) for i in range(3)]
                source_end = [pred_end_vx[i] - pred_start_vx[i] for i in range(3)]
                predictions[
                    :,
                    pred_start_vx[0] : pred_end_vx[0],
                    pred_start_vx[1] : pred_end_vx[1],
                    pred_start_vx[2] : pred_end_vx[2],
                ] = block_predictions[:, : source_end[0], : source_end[1], : source_end[2]]

    return predictions


def evaluate_single_channel(
    pred_channel: np.ndarray,
    gt_binary: np.ndarray,
    config,
) -> dict:
    """Evaluate a single prediction channel against ground truth using per-channel config."""
    all_scores = []
    best: Dict[str, float] = {}

    for pp in config.postprocessing:
        params, processed = pp.apply(pred_channel)
        metric_scores = {}

        if "dice" in config.metrics:
            metric_scores["dice"] = dice_score(gt_binary, processed)
        if "jaccard" in config.metrics:
            metric_scores["jaccard"] = jaccard(gt_binary, processed)

        all_scores.append({"postprocessing_type": pp.type, "params": params, "metrics": metric_scores})

        # Track best per metric
        for metric_name, score_val in metric_scores.items():
            best_key = f"best_{metric_name}"
            if score_val > best.get(best_key, 0.0):
                best[best_key] = score_val
                best[f"best_{metric_name}_params"] = params

    # Log key stats
    gt_positive_ratio = gt_binary.mean()
    best_strs = ", ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in best.items()
    )
    logger.info(
        f"{config.channel}: pred_range=[{pred_channel.min():.3f}, {pred_channel.max():.3f}], "
        f"pred_mean={pred_channel.mean():.3f}, gt_pos={gt_positive_ratio:.3f}, {best_strs}"
    )

    return {**best, "scores": all_scores}


@click.group()
def cli():
    pass


cli.add_command(query_group)


@cli.command()
@click.option(
    "--eval-config",
    "-e",
    type=click.Path(exists=True),
    required=True,
    help="Path to evaluation configuration YAML file",
)
@click.option(
    "--dataset",
    "-ds",
    type=str,
    required=False,
    help="Dataset name from data config to evaluate (default: all datasets)",
)
@click.option(
    "--crop",
    "-c",
    type=str,
    required=False,
    multiple=True,
    help="Specific crop(s) to evaluate (can be used multiple times, default: all crops)",
)
@click.option(
    "--checkpoint",
    "-ckpt",
    type=str,
    required=False,
    multiple=True,
    help="Specific checkpoint(s) to evaluate (can be used multiple times, default: all checkpoints in config)",
)
@click.option(
    "--db-url",
    type=str,
    required=False,
    help="Database URL for storing results (overrides config value, e.g. 'sqlite:///results.db')",
)
@click.option(
    "--save-predictions",
    type=click.Path(),
    required=False,
    help="Directory to save raw predictions as zarr (e.g. './predictions.zarr')",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
)
def run(
    eval_config: str,
    dataset: Optional[str],
    crop: tuple,
    checkpoint: tuple,
    db_url: Optional[str],
    save_predictions: Optional[str],
    log_level: str,
):
    """Evaluate model predictions on validation data."""
    pkg_logger = logging.getLogger("organelle_mapping")
    pkg_logger.setLevel(log_level.upper())

    # Load evaluation config
    logger.info(f"Loading evaluation config from {eval_config}")
    with open(eval_config) as f:
        eval_config_dict = yaml.safe_load(f)
    eval_cfg = TypeAdapter(EvaluationConfig).validate_python(eval_config_dict)

    # Get configuration components
    run_config = eval_cfg.experiment_run
    network_config = eval_cfg.eval_architecture
    sampling = run_config.sampling
    data_cfg = eval_cfg.data
    num_channels = run_config.total_channels

    # Build descriptor → model output channel index
    all_descriptors = run_config.channel_descriptors
    descriptor_to_index = {desc: i for i, desc in enumerate(all_descriptors)}

    # Build descriptor → source label (for GT loading)
    descriptor_to_source: Dict[str, str] = {}
    for target in run_config.targets:
        for transform in target.transforms:
            for desc in transform.channel_descriptors:
                descriptor_to_source[desc] = transform.source

    # Get eval channel configs (already defaulted + validated by EvaluationConfig)
    eval_channel_configs = list(eval_cfg.eval_channels)

    # Initialize database if configured (CLI flag overrides config)
    effective_db_url = db_url or eval_cfg.db_url
    db_engine = None
    if effective_db_url:
        db_engine = init_database(effective_db_url)

    # Compute channel indices and source labels for the active eval channels
    channel_indices = [descriptor_to_index[ec.channel] for ec in eval_channel_configs]
    eval_labels = [descriptor_to_source[ec.channel] for ec in eval_channel_configs]

    # Filter checkpoints if specific ones requested
    if checkpoint:
        requested_checkpoints = set(checkpoint)
        available_checkpoints = set(eval_cfg.checkpoints)
        if not requested_checkpoints.issubset(available_checkpoints):
            invalid_checkpoints = requested_checkpoints - available_checkpoints
            msg = f"Invalid checkpoints: {invalid_checkpoints}. Available: {eval_cfg.checkpoints}"
            raise ValueError(msg)
        checkpoints = [c for c in eval_cfg.checkpoints if c in requested_checkpoints]
        logger.info(f"Evaluating specific checkpoints: {checkpoints}")
    else:
        checkpoints = eval_cfg.checkpoints

    logger.info(f"Channels to evaluate: {[ec.channel for ec in eval_channel_configs]}")
    logger.info(f"Channel indices in model output: {channel_indices}")
    logger.info(f"Target sampling: {sampling}")
    logger.info(f"Architecture: {network_config.name} (padding={network_config.padding})")
    logger.info(f"Checkpoints to evaluate: {checkpoints}")

    # Determine which datasets and crops to evaluate
    if dataset:
        if dataset not in data_cfg.datasets:
            msg = f"Dataset '{dataset}' not found in data config"
            raise ValueError(msg)
        datasets_to_eval = {dataset: data_cfg.datasets[dataset]}
    else:
        datasets_to_eval = data_cfg.datasets
        logger.info(f"Evaluating all {len(datasets_to_eval)} datasets")

    # Filter crops if specific ones requested
    if crop:
        crops_to_eval = list(crop)
        logger.info(f"Evaluating specific crops: {crops_to_eval}")
    else:
        crops_to_eval = None

    # Store results for all checkpoints
    checkpoint_results = {}

    # Evaluate each checkpoint
    for ckpt in checkpoints:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Evaluating checkpoint: {ckpt}")
        logger.info(f"{'=' * 70}")

        # Load model
        logger.info(f"Loading model from {ckpt}")
        model = load_eval_model(network_config, run_config.targets, ckpt)
        device = next(model.parameters()).device

        all_results = {}

        # Iterate over datasets
        for ds_name, dataset_info in datasets_to_eval.items():
            logger.info(f"\nEvaluating dataset: {ds_name}")
            min_raw, max_raw = dataset_info.em.contrast

            # Determine crops for this dataset
            dataset_crops = []
            for crop_group in dataset_info.labels.crops:
                dataset_crops.extend(crop_group.split(","))

            if crops_to_eval is not None:
                crops_for_dataset = [c for c in crops_to_eval if c in dataset_crops]
                if not crops_for_dataset:
                    logger.info(f"None of the requested crops found in dataset '{ds_name}', skipping")
                    continue
                logger.info(f"Evaluating crops: {crops_for_dataset}")
            else:
                crops_for_dataset = dataset_crops
                logger.info(f"Evaluating {len(crops_for_dataset)} crops")

            # Evaluate each crop
            for crop_name in crops_for_dataset:
                logger.info(f"\nEvaluating crop: {crop_name}")
                try:
                    ckpt_path = Path(ckpt).resolve()
                    run_name = f"{ckpt_path.parent.parent.name}/{ckpt_path.parent.name}"
                    ckpt_name = ckpt_path.stem

                    # Build save path for predictions if requested
                    pred_save_path = None
                    if save_predictions:
                        pred_save_path = Path(save_predictions) / run_name / ckpt_name / ds_name / crop_name

                    results = evaluate_single_crop(
                        network_config=network_config,
                        model=model,
                        device=device,
                        dataset_info=dataset_info,
                        crop_name=crop_name,
                        eval_channel_configs=eval_channel_configs,
                        channel_indices=channel_indices,
                        eval_labels=eval_labels,
                        num_channels=num_channels,
                        sampling=sampling,
                        min_raw=min_raw,
                        max_raw=max_raw,
                        all_descriptors=all_descriptors,
                        save_path=pred_save_path,
                    )
                    all_results[f"{ds_name}/{crop_name}"] = results

                    # Write results to database
                    if db_engine is not None:
                        for channel_desc, channel_metrics in results.items():
                            for score_entry in channel_metrics["scores"]:
                                pp_type = score_entry["postprocessing_type"]
                                params = score_entry["params"]
                                for metric_name, score_val in score_entry["metrics"].items():
                                    insert_result(
                                        engine=db_engine,
                                        run=run_name,
                                        checkpoint=ckpt_name,
                                        dataset=ds_name,
                                        crop=crop_name,
                                        channel=channel_desc,
                                        label=descriptor_to_source[channel_desc],
                                        metric=metric_name,
                                        score=score_val,
                                        postprocessing_type=pp_type,
                                        threshold=params.get("threshold"),
                                    )
                        logger.info(f"Results for {ds_name}/{crop_name} written to database")
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Failed to evaluate {ds_name}/{crop_name}: {e}", exc_info=True)
                    continue

        # Store results for this checkpoint
        checkpoint_results[ckpt] = all_results

        # Log summary for this checkpoint
        logger.info(f"\n{'=' * 60}")
        logger.info(f"CHECKPOINT {ckpt} SUMMARY")
        logger.info(f"{'=' * 60}")

        if not all_results:
            logger.info("No crops were successfully evaluated.")
            continue

        # Collect per-channel scores
        channel_scores: Dict[str, Dict[str, list]] = {
            ec.channel: {f"best_{m}": [] for m in ec.metrics} for ec in eval_channel_configs
        }

        for _crop_id, crop_results in all_results.items():
            for channel_desc, metrics in crop_results.items():
                for metric_key in channel_scores.get(channel_desc, {}):
                    if metric_key in metrics:
                        channel_scores[channel_desc][metric_key].append(metrics[metric_key])

        # Log per-channel averages
        logger.info("Per-channel averages:")
        for channel_desc, metric_lists in channel_scores.items():
            parts = []
            for metric_key, values in metric_lists.items():
                if values:
                    parts.append(f"{metric_key}={np.mean(values):.4f} ± {np.std(values):.4f}")
            if parts:
                logger.info(f"  {channel_desc}: {', '.join(parts)}")

        logger.info(f"Total crops evaluated: {len(all_results)}")

    # Log final summary across all checkpoints
    logger.info(f"\n{'=' * 70}")
    logger.info("FINAL SUMMARY - ALL CHECKPOINTS")
    logger.info(f"{'=' * 70}")

    for ckpt_name, results in checkpoint_results.items():
        if results:
            # Aggregate first metric's best scores across all channels and crops
            all_scores = []
            for crop_results in results.values():
                for metrics in crop_results.values():
                    if "best_dice" in metrics:
                        all_scores.append(metrics["best_dice"])
            if all_scores:
                logger.info(f"\n{ckpt_name}:")
                logger.info(f"  Mean Dice: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
                logger.info(f"  Crops evaluated: {len(results)}")


def evaluate_single_crop(
    network_config,
    model,
    device,
    dataset_info,
    crop_name: str,
    eval_channel_configs: list,
    channel_indices: list,
    eval_labels: list,
    num_channels: int,
    sampling: dict,
    min_raw: float,
    max_raw: float,
    all_descriptors: Optional[list[str]] = None,
    save_path: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a single crop and return results.

    Args:
        eval_channel_configs: List of EvalChannelConfig for channels to evaluate.
        channel_indices: Model output channel index for each eval channel.
        eval_labels: Source label name for each eval channel (for GT loading).
        num_channels: Total number of model output channels.
        all_descriptors: Channel descriptor names for all model output channels (for saving).
        save_path: If provided, save raw predictions to this zarr path.
    """
    # Deduplicate labels while preserving order (multiple channels may share a source)
    unique_labels = list(dict.fromkeys(eval_labels))

    # Find the appropriate scale level based on sampling
    first_label = unique_labels[0]
    first_gt_path = f"{dataset_info.labels.data}/{dataset_info.labels.group}/{crop_name}/{first_label}"
    logger.info(f"Finding scale level for sampling {sampling} from: {first_gt_path}")

    # Open zarr group to find scale
    z = zarr.open(first_gt_path, mode="r")
    scale_level, _, resolution, _ = find_target_scale(z, sampling)
    logger.info(f"Using scale level: {scale_level} with resolution: {resolution}")

    voxel_size_coord = Coordinate(tuple(resolution.values()))

    # Open ground truth datasets for each unique source label
    gt_arrays = {}
    for label_name in unique_labels:
        gt_path = f"{dataset_info.labels.data}/{dataset_info.labels.group}/{crop_name}/{label_name}/{scale_level}"
        logger.info(f"Opening ground truth: {gt_path}")
        gt_xarray = fst.read_xarray(gt_path)

        # Get encoding and annotation type from attributes
        try:
            z = zarr.open(gt_path, mode="r")
            ann_attrs = z.attrs["cellmap"]["annotation"]
            encoding = ann_attrs["annotation_type"]["encoding"]
            ann_type = ann_attrs["annotation_type"]["type"]
            logger.info(f"Label {label_name} encoding: {encoding}, type: {ann_type}")
        except Exception as e:
            logger.warning(f"Could not get annotation attrs for {label_name}, using default: {e}")
            encoding = {"absent": 0, "present": 1, "unknown": 255}
            ann_type = "semantic_segmentation"

        # Convert to binary
        gt_array = gt_xarray.values
        if ann_type == "smooth_semantic_segmentation":
            # Smooth downsampled: continuous values, threshold at 0.5 (majority rule)
            gt_binary = ((gt_array >= 0.5 * encoding["present"]) & (gt_array != encoding["unknown"])).astype(np.uint8)
            logger.debug(
                f"{label_name}: Smooth GT, threshold 0.5 (min={gt_array.min():.3f}, max={gt_array.max():.3f})"
            )
        else:
            gt_binary = (gt_array == encoding["present"]).astype(np.uint8)

        gt_arrays[label_name] = (gt_xarray, gt_binary)

    # Get the crop's ROI from the first ground truth xarray
    first_gt_xarray = next(v[0] for v in gt_arrays.values())
    axes = list(first_gt_xarray.dims)
    start_coord = Coordinate(first_gt_xarray[0, 0, 0].coords[ax].item() for ax in axes)
    end_coord = Coordinate(first_gt_xarray[-1, -1, -1].coords[ax].item() for ax in axes)

    # Build gt_data array indexed by eval channel order
    logger.info("Loading ground truth data...")
    gt_data = np.stack([gt_arrays[lbl][1] for lbl in eval_labels])
    logger.info(f"GT data shape: {gt_data.shape}")
    logger.info(f"GT data has positive samples: {[gt_data[i].sum() > 0 for i in range(len(eval_labels))]}")

    # Open raw dataset and find appropriate scale level
    raw_path = f"{dataset_info.em.data}/{dataset_info.em.group}"
    logger.info(f"Opening raw data from: {raw_path}")

    z_raw = zarr.open(raw_path, mode="r")
    raw_scale_level, _, raw_resolution, _ = find_target_scale(z_raw, sampling)
    logger.info(f"Using raw scale level: {raw_scale_level} with resolution: {raw_resolution}")

    if raw_resolution != resolution:
        logger.warning(f"Raw resolution {raw_resolution} doesn't match GT resolution {resolution}")

    raw_scale_path = f"{raw_path}/{raw_scale_level}"
    logger.info(f"Opening raw dataset with fibsem_tools: {raw_scale_path}")
    raw_xarray = fst.read_xarray(raw_scale_path)

    logger.info(f"Checking if start_coord {start_coord} exists in raw coordinates...")
    logger.info(f"Raw coordinate ranges: {[(ax, raw_xarray.coords[ax].values[[0, -1]]) for ax in axes]}")

    max_resolution_attempts = 10
    for _attempt in range(max_resolution_attempts):
        if all(start_coord[dim] in raw_xarray.coords[ax] for dim, ax in enumerate(axes)):
            break
        logger.info("Coordinate mismatch detected, trying higher resolution...")
        higher_resolution = {ax: raw_resolution[ax] / 2.0 for ax in axes}
        logger.info(f"Looking for resolution: {higher_resolution}")
        raw_res_coord = Coordinate(raw_resolution[ax] for ax in axes)
        start_coord = start_coord - raw_res_coord / 4.0
        end_coord = end_coord + raw_res_coord / 4.0
        raw_scale_level, _, raw_resolution, _ = find_target_scale(z_raw, higher_resolution)
        raw_scale_path = f"{raw_path}/{raw_scale_level}"
        raw_xarray = fst.read_xarray(raw_scale_path)
        logger.info(f"Found {raw_scale_level} with resolution {raw_resolution}")
    else:
        msg = f"Could not find matching raw resolution after {max_resolution_attempts} attempts"
        raise RuntimeError(msg)

    # Predict all model channels using blockwise processing
    logger.info("Running blockwise prediction...")
    predictions = predict_crop(
        model=model,
        device=device,
        network_config=network_config,
        raw_xarray=raw_xarray,
        start_coord=start_coord,
        end_coord=end_coord,
        voxel_size_coord=voxel_size_coord,
        min_raw=min_raw,
        max_raw=max_raw,
        num_channels=num_channels,
    )

    # Save raw predictions if requested (one OME-NGFF group per channel)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        descriptors = all_descriptors or [str(i) for i in range(predictions.shape[0])]
        offset = list(start_coord)
        voxel_size = list(voxel_size_coord)
        spatial_axes = ["z", "y", "x"]

        for i, desc in enumerate(descriptors):
            channel_path = save_path / desc
            channel_path.mkdir(parents=True, exist_ok=True)
            group = zarr.open_group(str(channel_path), mode="w")
            group.array("s0", predictions[i], chunks=True, dtype="float32")
            group.attrs["multiscales"] = [
                {
                    "version": "0.4",
                    "axes": [{"name": ax, "type": "space", "unit": "nanometer"} for ax in spatial_axes],
                    "datasets": [
                        {
                            "path": "s0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": voxel_size},
                                {"type": "translation", "translation": offset},
                            ],
                        }
                    ],
                }
            ]
        logger.info(f"Saved raw predictions to {save_path}")

    # Extract only the channels we want to evaluate
    eval_predictions = predictions[channel_indices]

    if eval_predictions.shape[-3:] != gt_data.shape[-3:]:
        msg = f"Spatial shape mismatch: predictions {eval_predictions.shape[-3:]} vs GT {gt_data.shape[-3:]}"
        raise ValueError(msg)

    # Evaluate each channel with its own config
    logger.info("Evaluating predictions...")
    results = {}
    for i, ec in enumerate(eval_channel_configs):
        pred_channel = eval_predictions[i]
        gt_channel = gt_data[i]
        results[ec.channel] = evaluate_single_channel(pred_channel, gt_channel, ec)

    return results
