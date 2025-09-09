#!/usr/bin/env python3
"""
Script to save mito and er segmentations from exports.zarr as a properly formatted crop.
"""

import zarr
import numpy as np
import json

def main():
    # Open source data
    print("Loading source data...")
    ds = zarr.open("/nrs/saalfeld/heinrichl/data/cellmap_labels/finetuning/exports.zarr", "r")
    
    # Extract mito and er data
    print("Extracting mito and er data...")
    mito = np.array(ds["2025_08_14"]["merged_labels"]["s0"]) == 3
    er = np.array(ds["2025_08_14"]["merged_labels"]["s0"]) == 12672397
    
    proofread_ids = (
        4756895987,
        174082239,
        92293002,
        100669309,
        4756896106,
        251425816,
        329506582,
        4756895979
    )
    print(f"Setting proofread_ids: {proofread_ids}")

    # Get valid region
    print("Computing valid region...")
    valid = np.isin(np.array(ds["2025_08_14"]["neurons-padded"]["s0"]), proofread_ids)

    # Pad valid to match mito/er shape (only at the back/end)
    padding = tuple((0, mito.shape[ax] - valid.shape[ax]) for ax in range(mito.ndim))
    valid = np.pad(valid, padding, mode="constant", constant_values=False)
    
    # Convert to uint8 and mask invalid regions
    print("Converting to uint8 and masking invalid regions...")
    mito = mito.astype(np.uint8)
    mito[np.logical_not(valid)] = 255
    
    er = er.astype(np.uint8)
    er[np.logical_not(valid)] = 255
    
    # Find bounding box for valid regions (non-255 values)
    print("Finding bounding box for valid regions...")
    valid_mask = (mito != 255) | (er != 255)  # Union of valid regions
    coords = np.where(valid_mask)
    
    if len(coords[0]) == 0:
        raise ValueError("No valid (non-255) values found in data!")
    
    z_min, z_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    x_min, x_max = coords[2].min(), coords[2].max() + 1
    
    print(f"Bounding box: Z[{z_min}:{z_max}], Y[{y_min}:{y_max}], X[{x_min}:{x_max}]")
    print(f"Size: ({z_max-z_min}, {y_max-y_min}, {x_max-x_min})")
    
    # Crop data to bounding box
    mito_cropped = mito[z_min:z_max, y_min:y_max, x_min:x_max]
    er_cropped = er[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Create target zarr file with proper hierarchy
    print("Creating target zarr file...")
    tgt_grp = zarr.open("/nrs/saalfeld/heinrichl/data/cellmap_labels/finetuning/jrc_fly-vnc-1/jrc_fly-vnc-1.zarr", "w")
    
    # Create group hierarchy
    recon_grp = tgt_grp.create_group("recon-1", overwrite=True)
    labels_grp = recon_grp.create_group("labels", overwrite=True)
    groundtruth_grp = labels_grp.create_group("groundtruth", overwrite=True)
    
    # Create crop group
    crop_grp = groundtruth_grp.create_group("crop_2025_08_14", overwrite=True)
    
    # Add metadata to the crop group
    print("Adding crop metadata...")
    crop_grp.attrs.update({
        "cellmap": {
            "annotation": {
                "class_names": [
                    "mito",
                    "er"
                ],
                "created_by": [],
                "created_with": [],
                "description": "Finetuning crop extracted from 2025_06_26 export",
                "duration_days": None,
                "end_date": None,
                "name": "crop_2025_06_26",
                "protocol_uri": None,
                "start_date": None,
                "version": "0.1.1"
            }
        }
    })
    
    # Resolution and offset (accounting for cropping)
    resolution = [8.0, 8.0, 8.0]  # nm
    # Adjust offset for the cropped region
    offset = [2.0 + z_min * 8.0, 2.0 + y_min * 8.0, 2.0 + x_min * 8.0]  # nm
    
    # Create mito group
    print("Saving mito data...")
    mito_grp = crop_grp.create_group("mito", overwrite=True)
    
    # Create the s0 dataset for mito
    mito_s0 = mito_grp.create_dataset(
        "s0",
        data=mito_cropped,
        chunks=(128, 128, 128),
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2),
        overwrite=True
    )
    
    # Add metadata for mito group
    mito_grp.attrs.update({
        "cellmap": {
            "annotation": {
                "annotation_type": {
                    "encoding": {
                        "absent": 0,
                        "present": 1,
                        "unknown": 255
                    },
                    "type": "semantic_segmentation"
                },
                "class_name": "mito"
            }
        },
        "multiscales": [{
            "axes": [
                {"name": "z", "type": "space", "unit": "nanometer"},
                {"name": "y", "type": "space", "unit": "nanometer"},
                {"name": "x", "type": "space", "unit": "nanometer"}
            ],
            "datasets": [{
                "coordinateTransformations": [
                    {"scale": resolution, "type": "scale"},
                    {"translation": offset, "type": "translation"}
                ],
                "path": "s0"
            }],
            "name": "nominal",
            "version": "0.4"
        }]
    })
    
    # Add metadata for mito s0 dataset
    mito_s0.attrs.update({
        "cellmap": {
            "annotation": {
                "annotation_type": {
                    "encoding": {
                        "absent": 0,
                        "present": 1,
                        "unknown": 255
                    },
                    "type": "semantic_segmentation"
                },
                "class_name": "mito"
            }
        }
    })
    
    # Create er group
    print("Saving ER data...")
    er_grp = crop_grp.create_group("er", overwrite=True)
    
    # Create the s0 dataset for er
    er_s0 = er_grp.create_dataset(
        "s0",
        data=er_cropped,
        chunks=(128, 128, 128),
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2),
        overwrite=True
    )
    
    # Add metadata for er group
    er_grp.attrs.update({
        "cellmap": {
            "annotation": {
                "annotation_type": {
                    "encoding": {
                        "absent": 0,
                        "present": 1,
                        "unknown": 255
                    },
                    "type": "semantic_segmentation"
                },
                "class_name": "er"
            }
        },
        "multiscales": [{
            "axes": [
                {"name": "z", "type": "space", "unit": "nanometer"},
                {"name": "y", "type": "space", "unit": "nanometer"},
                {"name": "x", "type": "space", "unit": "nanometer"}
            ],
            "datasets": [{
                "coordinateTransformations": [
                    {"scale": resolution, "type": "scale"},
                    {"translation": offset, "type": "translation"}
                ],
                "path": "s0"
            }],
            "name": "nominal",
            "version": "0.4"
        }]
    })
    
    # Add metadata for er s0 dataset
    er_s0.attrs.update({
        "cellmap": {
            "annotation": {
                "annotation_type": {
                    "encoding": {
                        "absent": 0,
                        "present": 1,
                        "unknown": 255
                    },
                    "type": "semantic_segmentation"
                },
                "class_name": "er"
            }
        }
    })
    
    # Verify the saved data
    print("\nVerification:")
    print(f"Original shape: {mito.shape}")
    print(f"Cropped Mito shape: {mito_grp['s0'].shape}")
    print(f"Cropped ER shape: {er_grp['s0'].shape}")
    print(f"Mito unique values: {np.unique(mito_grp['s0'][:])}")
    print(f"ER unique values: {np.unique(er_grp['s0'][:])}")
    print(f"Adjusted offset: {offset}")
    print(f"\nSaved to: /nrs/saalfeld/heinrichl/data/cellmap_labels/finetuning/jrc_fly-vnc-1/jrc_fly-vnc-1.zarr/recon-1/labels/groundtruth/crop_2025_08_14/")
    print("Done!")

if __name__ == "__main__":
    main()