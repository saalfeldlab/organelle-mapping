#!/usr/bin/env python3
"""
Script to save mito and er segmentations from exports.zarr as a properly formatted crop.
"""

import logging

import numpy as np
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MITO_LABEL_ID = 3
ER_LABEL_ID = 12672397
UNKNOWN_VALUE = 255


def main():
    # Open source data
    logger.info("Loading source data...")
    ds = zarr.open("/nrs/saalfeld/heinrichl/data/cellmap_labels/finetuning/exports.zarr", "r")

    # Extract mito and er data
    logger.info("Extracting mito and er data...")
    mito = np.array(ds["2025_08_14"]["merged_labels"]["s0"]) == MITO_LABEL_ID
    er = np.array(ds["2025_08_14"]["merged_labels"]["s0"]) == ER_LABEL_ID

    proofread_ids = (4756895987, 174082239, 92293002, 100669309, 4756896106, 251425816, 329506582, 4756895979)
    logger.info(f"Setting proofread_ids: {proofread_ids}")

    # Get valid region
    logger.info("Computing valid region...")
    valid = np.isin(np.array(ds["2025_08_14"]["neurons-padded"]["s0"]), proofread_ids)

    # Pad valid to match mito/er shape (only at the back/end)
    padding = tuple((0, mito.shape[ax] - valid.shape[ax]) for ax in range(mito.ndim))
    valid = np.pad(valid, padding, mode="constant", constant_values=False)

    # Convert to uint8 and mask invalid regions
    logger.info("Converting to uint8 and masking invalid regions...")
    mito = mito.astype(np.uint8)
    mito[np.logical_not(valid)] = UNKNOWN_VALUE

    er = er.astype(np.uint8)
    er[np.logical_not(valid)] = UNKNOWN_VALUE

    # Find bounding box for valid regions (non-UNKNOWN_VALUE values)
    logger.info("Finding bounding box for valid regions...")
    valid_mask = (mito != UNKNOWN_VALUE) | (er != UNKNOWN_VALUE)  # Union of valid regions
    coords = np.where(valid_mask)

    if len(coords[0]) == 0:
        msg = "No valid (non-255) values found in data!"
        raise ValueError(msg)

    z_min, z_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    x_min, x_max = coords[2].min(), coords[2].max() + 1

    logger.info(f"Bounding box: Z[{z_min}:{z_max}], Y[{y_min}:{y_max}], X[{x_min}:{x_max}]")
    logger.info(f"Size: ({z_max - z_min}, {y_max - y_min}, {x_max - x_min})")

    # Crop data to bounding box
    mito_cropped = mito[z_min:z_max, y_min:y_max, x_min:x_max]
    er_cropped = er[z_min:z_max, y_min:y_max, x_min:x_max]

    # Create target zarr file with proper hierarchy
    logger.info("Creating target zarr file...")
    tgt_grp = zarr.open("/nrs/saalfeld/heinrichl/data/cellmap_labels/finetuning/jrc_fly-vnc-1/jrc_fly-vnc-1.zarr", "w")

    # Create group hierarchy
    recon_grp = tgt_grp.create_group("recon-1", overwrite=True)
    labels_grp = recon_grp.create_group("labels", overwrite=True)
    groundtruth_grp = labels_grp.create_group("groundtruth", overwrite=True)

    # Create crop group
    crop_grp = groundtruth_grp.create_group("crop_2025_08_14", overwrite=True)

    # Add metadata to the crop group
    logger.info("Adding crop metadata...")
    crop_grp.attrs.update(
        {
            "cellmap": {
                "annotation": {
                    "class_names": ["mito", "er"],
                    "created_by": [],
                    "created_with": [],
                    "description": "Finetuning crop extracted from 2025_06_26 export",
                    "duration_days": None,
                    "end_date": None,
                    "name": "crop_2025_06_26",
                    "protocol_uri": None,
                    "start_date": None,
                    "version": "0.1.1",
                }
            }
        }
    )

    # Resolution and offset (accounting for cropping)
    resolution = [8.0, 8.0, 8.0]  # nm
    # Adjust offset for the cropped region
    offset = [2.0 + z_min * 8.0, 2.0 + y_min * 8.0, 2.0 + x_min * 8.0]  # nm

    # Create mito group
    logger.info("Saving mito data...")
    mito_grp = crop_grp.create_group("mito", overwrite=True)

    # Create the s0 dataset for mito
    mito_s0 = mito_grp.create_dataset(
        "s0",
        data=mito_cropped,
        chunks=(128, 128, 128),
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
        overwrite=True,
    )

    # Add metadata for mito group
    mito_grp.attrs.update(
        {
            "cellmap": {
                "annotation": {
                    "annotation_type": {
                        "encoding": {"absent": 0, "present": 1, "unknown": 255},
                        "type": "semantic_segmentation",
                    },
                    "class_name": "mito",
                }
            },
            "multiscales": [
                {
                    "axes": [
                        {"name": "z", "type": "space", "unit": "nanometer"},
                        {"name": "y", "type": "space", "unit": "nanometer"},
                        {"name": "x", "type": "space", "unit": "nanometer"},
                    ],
                    "datasets": [
                        {
                            "coordinateTransformations": [
                                {"scale": resolution, "type": "scale"},
                                {"translation": offset, "type": "translation"},
                            ],
                            "path": "s0",
                        }
                    ],
                    "name": "nominal",
                    "version": "0.4",
                }
            ],
        }
    )

    # Add metadata for mito s0 dataset
    mito_s0.attrs.update(
        {
            "cellmap": {
                "annotation": {
                    "annotation_type": {
                        "encoding": {"absent": 0, "present": 1, "unknown": 255},
                        "type": "semantic_segmentation",
                    },
                    "class_name": "mito",
                }
            }
        }
    )

    # Create er group
    logger.info("Saving ER data...")
    er_grp = crop_grp.create_group("er", overwrite=True)

    # Create the s0 dataset for er
    er_s0 = er_grp.create_dataset(
        "s0",
        data=er_cropped,
        chunks=(128, 128, 128),
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
        overwrite=True,
    )

    # Add metadata for er group
    er_grp.attrs.update(
        {
            "cellmap": {
                "annotation": {
                    "annotation_type": {
                        "encoding": {"absent": 0, "present": 1, "unknown": 255},
                        "type": "semantic_segmentation",
                    },
                    "class_name": "er",
                }
            },
            "multiscales": [
                {
                    "axes": [
                        {"name": "z", "type": "space", "unit": "nanometer"},
                        {"name": "y", "type": "space", "unit": "nanometer"},
                        {"name": "x", "type": "space", "unit": "nanometer"},
                    ],
                    "datasets": [
                        {
                            "coordinateTransformations": [
                                {"scale": resolution, "type": "scale"},
                                {"translation": offset, "type": "translation"},
                            ],
                            "path": "s0",
                        }
                    ],
                    "name": "nominal",
                    "version": "0.4",
                }
            ],
        }
    )

    # Add metadata for er s0 dataset
    er_s0.attrs.update(
        {
            "cellmap": {
                "annotation": {
                    "annotation_type": {
                        "encoding": {"absent": 0, "present": 1, "unknown": 255},
                        "type": "semantic_segmentation",
                    },
                    "class_name": "er",
                }
            }
        }
    )

    # Verify the saved data
    logger.info("Verification:")
    logger.info(f"Original shape: {mito.shape}")
    logger.info(f"Cropped Mito shape: {mito_grp['s0'].shape}")
    logger.info(f"Cropped ER shape: {er_grp['s0'].shape}")
    logger.info(f"Mito unique values: {np.unique(mito_grp['s0'][:])}")
    logger.info(f"ER unique values: {np.unique(er_grp['s0'][:])}")
    logger.info(f"Adjusted offset: {offset}")
    tgt_path = tgt_grp.store.path
    logger.info(f"Saved to: {tgt_path}/recon-1/labels/groundtruth/crop_2025_08_14/")
    logger.info("Done!")



if __name__ == "__main__":
    main()
