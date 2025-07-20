import numpy as np
import torch
import yaml
import zarr
from typing import Dict, Optional
import logging
import click
import fibsem_tools as fst
from funlib.geometry.coordinate import Coordinate
from funlib.geometry.roi import Roi
from funlib.persistence import open_ds
from pydantic import TypeAdapter

from organelle_mapping.config.run import RunConfig
from organelle_mapping.config.data import DataConfig
from organelle_mapping.model import load_eval_model
from organelle_mapping.metrics import dice_score, jaccard
from organelle_mapping.utils import find_target_scale, get_axes_names
from skimage.measure import block_reduce

logger = logging.getLogger("organelle_mapping.evaluation")


def normalize_raw(raw: np.ndarray, min_val: float = 0, max_val: float = 255) -> np.ndarray:
    """Normalize raw data to match training normalization."""
    # Match training: gp.Normalize(factor=1/factor) + gp.IntensityScaleShift
    # Determine factor based on data type (like training does)
    factors = {np.dtype("uint8"): 255, np.dtype("uint16"): 2**16 - 1}
    factor = factors.get(raw.dtype, 255.0)  # default to 255 if unknown
    
    normalized = raw.astype(np.float32) / factor  # [0, 1] range
    scale = (max_val - min_val) / factor
    shift = min_val / factor
    return normalized * scale + shift


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
    label_list: list,
) -> np.ndarray:
    """Predict on an entire crop using blockwise processing."""
    logger.info(f"Start coord {start_coord}, end coord {end_coord}")
    block_input_shape = Coordinate(network_config.input_shape)
    block_output_shape = Coordinate(network_config.output_shape)
    block_input_shape_world = block_input_shape * voxel_size_coord
    block_output_shape_world = block_output_shape * voxel_size_coord
    
    context = (block_input_shape_world - block_output_shape_world) / 2.
    raw_begin = Coordinate(raw_xarray[0,0,0].coords[ax].item() for ax in raw_xarray.dims)
    raw_end = Coordinate(raw_xarray[-1,-1,-1].coords[ax].item() for ax in raw_xarray.dims)
    raw_resolution = Coordinate(raw_xarray[1,1,1].coords[ax].item() for ax in raw_xarray.dims) - raw_begin
    shape = ((end_coord - start_coord + raw_resolution) / voxel_size_coord)
    logging.info(f"Calculatd shape as {shape}")
    # Initialize prediction array matching GT size
    predictions = np.zeros((len(label_list),) + tuple(shape), dtype=np.float32)
    
    # Process in blocks
    for z_start in range(start_coord[0], end_coord[0], block_output_shape_world[0]):
        for y_start in range(start_coord[1], end_coord[1], block_output_shape_world[1]):
            for x_start in range(start_coord[2], end_coord[2], block_output_shape_world[2]):

                # Calculate world-space ROI for this block  
                block_begin_world = Coordinate((z_start, y_start, x_start))
                block_end_world = block_begin_world + block_input_shape_world
                # Calculate world-space input ROI for this block
                input_begin_world = block_begin_world - context
                input_end_world = input_begin_world + block_input_shape_world - raw_resolution
                begin_oob = [s - r for s,r in zip(input_begin_world, raw_begin)]
                end_oob = [r - s for s,r in zip(input_end_world, raw_end)]
                
                # Create slices for xarray
                slices = tuple(slice(s, e) for s, e in zip(input_begin_world, input_end_world))
                # Read from xarray
                block_raw = raw_xarray.loc[slices].values
                logger.debug(f"Read raw data of shape {block_raw.shape}")
                if raw_resolution != voxel_size_coord:
                    downsample_factor = tuple(int(voxel_size_coord[i] / raw_resolution[i]) for i in range(3))
                    block_raw = block_reduce(block_raw, downsample_factor,func=np.mean)
                
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
                    block_raw = np.pad(block_raw, pad_width, mode='constant', constant_values=min_raw)
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
                pred_start_vx = (block_begin_world - start_coord)/voxel_size_coord
                pred_end_vx = pred_start_vx + block_output_shape
                pred_end_vx = [min(pred_end_vx[i], shape[i]) for i in range(3)]
                source_end = [pred_end_vx[i] - pred_start_vx[i] for i in range(3)]
                predictions[
                    :,
                    pred_start_vx[0]:pred_end_vx[0],
                    pred_start_vx[1]:pred_end_vx[1],
                    pred_start_vx[2]:pred_end_vx[2]
                ] = block_predictions[:, :source_end[0], :source_end[1], :source_end[2]]
    
    return predictions


def evaluate_predictions(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    channel_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate predictions against ground truth for all channels with threshold sweep."""
    assert predictions.shape == ground_truth.shape, (
        f"Shape mismatch: predictions {predictions.shape} vs ground_truth {ground_truth.shape}"
    )
    
    n_channels = predictions.shape[0]
    results = {}
    
    # Define thresholds to sweep
    thresholds = np.linspace(0.01, 0.5, 50)
    
    for channel in range(n_channels):
        pred_channel = predictions[channel]
        gt_channel = ground_truth[channel]
        gt_binary = gt_channel.astype(np.uint8)
        
        channel_name = channel_names.get(channel, f"channel_{channel}") if channel_names else f"channel_{channel}"
        
        # Sweep thresholds to find best scores
        best_dice = 0.0
        best_dice_threshold = 0.0
        best_jaccard = 0.0
        best_jaccard_threshold = 0.0
        
        for threshold in thresholds:
            pred_binary = (pred_channel > threshold).astype(np.uint8)
            
            dice = dice_score(gt_binary, pred_binary)
            jacc = jaccard(gt_binary, pred_binary)
            
            if dice > best_dice:
                best_dice = dice
                best_dice_threshold = threshold
            
            if jacc > best_jaccard:
                best_jaccard = jacc
                best_jaccard_threshold = threshold
        
        # Show key stats
        gt_positive_ratio = gt_binary.mean()
        pred_positive_ratio = (pred_channel > best_dice_threshold).mean()
        logger.info(f"{channel_name}: pred_range=[{pred_channel.min():.3f}, {pred_channel.max():.3f}], "
                   f"pred_mean={pred_channel.mean():.3f}, gt_pos={gt_positive_ratio:.3f}, "
                   f"pred_pos={pred_positive_ratio:.3f}, "
                   f"best_dice_thr={best_dice_threshold:.3f}, best_jacc_thr={best_jaccard_threshold:.3f}")
        
        results[channel_name] = {
            "dice": best_dice,
            "dice_threshold": best_dice_threshold,
            "jaccard": best_jaccard,
            "jaccard_threshold": best_jaccard_threshold,
        }
    
    return results


@click.command()
@click.option(
    "--run-config",
    "-r",
    type=click.Path(exists=True),
    required=True,
    help="Path to run configuration YAML file",
)
@click.option(
    "--checkpoint",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to model checkpoint",
)
@click.option(
    "--data-config",
    "-d",
    type=click.Path(exists=True),
    required=False,
    help="Path to data configuration YAML file (overrides run config's data)",
)
@click.option(
    "--dataset-name",
    "-dn",
    type=str,
    required=False,
    help="Dataset name from data config to evaluate (default: all datasets)",
)
@click.option(
    "--crop-name",
    "-cn",
    type=str,
    required=False,
    help="Name of the crop to evaluate (default: all crops)",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
)
def evaluate(
    run_config: str,
    checkpoint: str,
    data_config: Optional[str],
    dataset_name: Optional[str],
    crop_name: Optional[str],
    log_level: str,
):
    """Evaluate model predictions on a validation block."""
    pkg_logger = logging.getLogger("organelle_mapping")
    pkg_logger.setLevel(log_level.upper())
    
    # Load run config
    logger.info(f"Loading run config from {run_config}")
    with open(run_config) as f:
        run_config_dict = yaml.safe_load(f)
    config = TypeAdapter(RunConfig).validate_python(run_config_dict)
    
    # Get architecture and labels from run config
    network_config = config.architecture
    label_list = list(config.labels)
    sampling = config.sampling
    logger.info(f"Labels to evaluate: {label_list}")
    logger.info(f"Target sampling: {sampling}")
    
    # Load model
    logger.info(f"Loading model from {checkpoint}")
    model = load_eval_model(network_config, checkpoint)
    device = next(model.parameters()).device
    
    # Load data config if provided, otherwise use the one from run config
    if data_config:
        logger.info(f"Loading data config from {data_config}")
        with open(data_config) as f:
            data_dict = yaml.safe_load(f)
        data_cfg = TypeAdapter(DataConfig).validate_python(data_dict)
    else:
        logger.info("Using data config from run config")
        data_cfg = config.data
    
    # Determine which datasets and crops to evaluate
    if dataset_name:
        if dataset_name not in data_cfg.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in data config")
        datasets_to_eval = {dataset_name: data_cfg.datasets[dataset_name]}
    else:
        datasets_to_eval = data_cfg.datasets
        logger.info(f"Evaluating all {len(datasets_to_eval)} datasets")
    
    # Store results for all evaluations
    all_results = {}
    
    # Iterate over datasets
    for ds_name, dataset_info in datasets_to_eval.items():
        logger.info(f"\nEvaluating dataset: {ds_name}")
        
        # Get contrast values for normalization
        min_raw, max_raw = dataset_info.em.contrast
        
        # Determine which crops to evaluate
        if crop_name:
            # Check if crop exists
            crop_found = False
            for crop_group in dataset_info.labels.crops:
                if crop_name in crop_group.split(","):
                    crops_to_eval = [crop_name]
                    crop_found = True
                    break
            if not crop_found:
                logger.warning(f"Crop '{crop_name}' not found in dataset '{ds_name}', skipping")
                continue
        else:
            # Get all crops
            crops_to_eval = []
            for crop_group in dataset_info.labels.crops:
                crops_to_eval.extend(crop_group.split(","))
            logger.info(f"Evaluating {len(crops_to_eval)} crops")
        
        # Evaluate each crop
        for crop in crops_to_eval:
            logger.info(f"\nEvaluating crop: {crop}")
            try:
                results = evaluate_single_crop(
                    network_config=network_config,
                    model=model,
                    device=device,
                    dataset_info=dataset_info,
                    crop_name=crop,
                    label_list=label_list,
                    sampling=sampling,
                    min_raw=min_raw,
                    max_raw=max_raw,
                )
                all_results[f"{ds_name}/{crop}"] = results
            except Exception as e:
                logger.error(f"Failed to evaluate {ds_name}/{crop}: {e}", exc_info=True)
                continue
    
    # Print summary of all results
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    if not all_results:
        print("No crops were successfully evaluated.")
        return
    
    # Calculate overall statistics
    all_dice_scores = []
    all_jaccard_scores = []
    label_dice_scores = {label: [] for label in label_list}
    label_jaccard_scores = {label: [] for label in label_list}
    
    for crop_id, crop_results in all_results.items():
        print(f"\n{crop_id}:")
        for label, metrics in crop_results.items():
            print(f"  {label}: Dice={metrics['dice']:.4f} (thr={metrics['dice_threshold']:.3f}), "
                  f"Jaccard={metrics['jaccard']:.4f} (thr={metrics['jaccard_threshold']:.3f})")
            all_dice_scores.append(metrics['dice'])
            all_jaccard_scores.append(metrics['jaccard'])
            label_dice_scores[label].append(metrics['dice'])
            label_jaccard_scores[label].append(metrics['jaccard'])
    
    # Print per-label averages
    print("\n" + "-" * 50)
    print("Per-label averages:")
    for label in label_list:
        if label_dice_scores[label]:
            avg_dice = np.mean(label_dice_scores[label])
            avg_jaccard = np.mean(label_jaccard_scores[label])
            std_dice = np.std(label_dice_scores[label])
            std_jaccard = np.std(label_jaccard_scores[label])
            print(f"  {label}:")
            print(f"    Dice:    {avg_dice:.4f} ± {std_dice:.4f}")
            print(f"    Jaccard: {avg_jaccard:.4f} ± {std_jaccard:.4f}")
    
    # Print overall averages
    print("\n" + "-" * 50)
    print("Overall averages:")
    print(f"  Mean Dice Score:    {np.mean(all_dice_scores):.4f} ± {np.std(all_dice_scores):.4f}")
    print(f"  Mean Jaccard Score: {np.mean(all_jaccard_scores):.4f} ± {np.std(all_jaccard_scores):.4f}")
    print(f"  Total crops evaluated: {len(all_results)}")
    print("=" * 70)


def evaluate_single_crop(
    network_config,
    model,
    device,
    dataset_info,
    crop_name: str,
    label_list: list,
    sampling: dict,
    min_raw: float,
    max_raw: float,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a single crop and return results."""
    
    # Find the appropriate scale level based on sampling
    first_label = label_list[0]
    first_gt_path = f"{dataset_info.labels.data}/{dataset_info.labels.group}/{crop_name}/{first_label}"
    logger.info(f"Finding scale level for sampling {sampling} from: {first_gt_path}")
    
    # Open zarr group to find scale
    z = zarr.open(first_gt_path, mode='r')
    scale_level, _, resolution, _ = find_target_scale(z, sampling)
    logger.info(f"Using scale level: {scale_level} with resolution: {resolution}")
    
    voxel_size_coord = Coordinate(tuple(resolution.values()))
    
    # Open ground truth datasets for each label
    gt_xarrays = {}
    gt_encodings = {}
    for i, label in enumerate(label_list):
        gt_path = f"{dataset_info.labels.data}/{dataset_info.labels.group}/{crop_name}/{label}/{scale_level}"
        logger.info(f"Opening ground truth: {gt_path}")
        gt_xarray = fst.read_xarray(gt_path)
        gt_xarrays[label] = gt_xarray
        
        # Get encoding from attributes
        try:
            z = zarr.open(gt_path, mode='r')
            encoding = z.attrs["cellmap"]["annotation"]["annotation_type"]["encoding"]
            gt_encodings[label] = encoding
            logger.info(f"Label {label} encoding: {encoding}")
        except Exception as e:
            logger.warning(f"Could not get encoding for {label}, using default: {e}")
            gt_encodings[label] = {"absent": 0, "present": 1, "unknown": 255}
    
    # Get the crop's ROI from the first ground truth xarray
    # All labels in the same crop should have the same ROI
    first_gt_xarray = next(iter(gt_xarrays.values()))
    axes = list(first_gt_xarray.dims)  # Get axes from xarray dimensions instead
    # Extract world coordinates from xarray
    start_coord = Coordinate(first_gt_xarray[0, 0, 0].coords[ax].item() for ax in axes)

    # Calculate ROI from coordinates
    end_coord = Coordinate(first_gt_xarray[-1, -1, -1].coords[ax].item() for ax in axes)
    
    # Verify all GT xarrays have the same shape
    for label, gt_xarray in gt_xarrays.items():
        if gt_xarray.shape != first_gt_xarray.shape:
            logger.warning(f"Label {label} has different shape: {gt_xarray.shape} vs {first_gt_xarray.shape}")

    logger.info("Loading ground truth data...")
    gt_data = np.zeros((len(label_list),) + tuple(first_gt_xarray.shape), dtype=np.uint8)
    for i, (label, gt_xarray) in enumerate(gt_xarrays.items()):
        # Read the entire GT array (it's already cropped to the annotation region)
        gt_array = gt_xarray.values
        # Convert to binary: present=1, absent/unknown=0
        encoding = gt_encodings[label]
        # Use threshold approach (handles both exact and downsampled cases)
        if encoding["present"] == 1:
            gt_binary = (gt_array > 0).astype(np.uint8)
            logger.debug(f"{label}: Using threshold 0 for GT (min={gt_array.min():.3f}, max={gt_array.max():.3f})")
        else:
            # For other encoding schemes, use exact match
            gt_binary = (gt_array == encoding["present"]).astype(np.uint8)
        gt_data[i] = gt_binary
    
    # Open raw dataset and find appropriate scale level
    raw_path = f"{dataset_info.em.data}/{dataset_info.em.group}"
    logger.info(f"Opening raw data from: {raw_path}")
    
    # Use find_target_scale for raw data too
    z_raw = zarr.open(raw_path, mode='r')
    raw_scale_level, _, raw_resolution, _ = find_target_scale(z_raw, sampling)
    logger.info(f"Using raw scale level: {raw_scale_level} with resolution: {raw_resolution}")
    
    # Verify resolution matches
    if raw_resolution != resolution:
        logger.warning(f"Raw resolution {raw_resolution} doesn't match GT resolution {resolution}")
    
    # Open the raw dataset at the found scale using fibsem_tools
    raw_scale_path = f"{raw_path}/{raw_scale_level}"
    logger.info(f"Opening raw dataset with fibsem_tools: {raw_scale_path}")
    raw_xarray = fst.read_xarray(raw_scale_path)
    
    
    logger.info(f"Checking if start_coord {start_coord} exists in raw coordinates...")
    logger.info(f"Raw coordinate ranges: {[(ax, raw_xarray.coords[ax].values[[0, -1]]) for ax in axes]}")
    
    while any(start_coord[dim] not in raw_xarray.coords[ax] for dim, ax in enumerate(axes)):
        logger.info(f"Coordinate mismatch detected, trying higher resolution...")
        higher_resolution = {ax: raw_resolution[ax]/2. for ax in axes}
        logger.info(f"Looking for resolution: {higher_resolution}")
        # Convert resolution dict to Coordinate for arithmetic
        raw_res_coord = Coordinate(raw_resolution[ax] for ax in axes)
        start_coord = start_coord - raw_res_coord/4.
        end_coord = end_coord + raw_res_coord/4.
        raw_scale_level, _, raw_resolution, _ = find_target_scale(z_raw, higher_resolution)
        raw_scale_path = f"{raw_path}/{raw_scale_level}"
        raw_xarray = fst.read_xarray(raw_scale_path)
        
        logger.info(f"Found {raw_scale_level} with resolution {raw_resolution}")

    # Predict using blockwise processing
    logger.info("Running blockwise prediction...")
    logger.info(f"GT data shape: {gt_data.shape}, dtype: {gt_data.dtype}")
    logger.info(f"GT data has positive samples: {[gt_data[i].sum() > 0 for i in range(len(label_list))]}")
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
        label_list=label_list,
    )

    # Crop predictions to match GT shape if needed
    if predictions.shape[-3:] != gt_data.shape[-3:]:
        msg = "Spatial shape of predictions does not match groundtruth."
        raise ValueError(msg)    
    # Evaluate
    logger.info("Evaluating predictions...")
    # Create channel name map from label names
    channel_name_map = {i: label for i, label in enumerate(label_list)}
    results = evaluate_predictions(predictions, gt_data, channel_name_map)
    
    return results




if __name__ == "__main__":
    evaluate()
