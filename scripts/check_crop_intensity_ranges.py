#!/usr/bin/env python
"""Generate intensity histograms for EM crops in cellmap datasets."""

import csv
import sys

import click
import numpy as np
import zarr

from organelle_mapping import utils
from organelle_mapping.config import DataConfig
from organelle_mapping.config.run import load_subconfig


@click.command()
@click.argument("data_config", type=click.Path(exists=True))
@click.option("--dataset", "-d", help="Specific dataset to check (default: all)")
@click.option("--output", "-o", default="histograms.csv", help="Output CSV file")
@click.option("--sampling", type=float, default=8.0, help="Target voxel size in nm (default: 8.0)")
def main(data_config, dataset, output, sampling):
    """Generate intensity histograms for EM crops in cellmap datasets.

    Outputs CSV with columns: dataset, crop, intensity, count
    """
    config = load_subconfig(data_config, DataConfig)
    target_resolution = {"z": sampling, "y": sampling, "x": sampling}

    datasets_to_check = [dataset] if dataset else sorted(config.datasets.keys())

    with open(output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["dataset", "crop", "intensity", "count"])

        for dataset_name in datasets_to_check:
            if dataset_name not in config.datasets:
                click.echo(f"ERROR: Dataset {dataset_name} not found in config", err=True)
                continue

            ds_info = config.datasets[dataset_name]

            try:
                # Open EM data and find target scale
                em_path = f"{ds_info.em.data}/{ds_info.em.group}"
                em_grp = zarr.open(em_path, "r")

                scale_name, em_offset_dict, em_resolution_dict, em_shape_dict = utils.find_target_scale(
                    em_grp, target_resolution
                )
                em_data = em_grp[scale_name]
                em_resolution = utils.ax_dict_to_list(em_resolution_dict, ["z", "y", "x"])

                # Open labels
                labels_path = f"{ds_info.labels.data}/{ds_info.labels.group}"
                labels_grp = zarr.open(labels_path, "r")

                # Iterate over crops specified in config
                for crops_str in ds_info.labels.crops:
                    for crop_name in crops_str.split(","):
                        crop_name = crop_name.strip()

                        if crop_name not in labels_grp:
                            click.echo(f"WARNING: Crop {crop_name} not found in {dataset_name}", err=True)
                            continue

                        crop_grp = labels_grp[crop_name]

                        if "organelle" not in crop_grp:
                            click.echo(f"WARNING: No organelle group in {dataset_name}/{crop_name}", err=True)
                            continue

                        org_grp = crop_grp["organelle"]

                        # Find target scale for crop
                        crop_scale_name, crop_offset_dict, crop_resolution_dict, crop_shape_dict = (
                            utils.find_target_scale(org_grp, target_resolution)
                        )

                        crop_offset = utils.ax_dict_to_list(crop_offset_dict, ["z", "y", "x"])
                        crop_resolution = utils.ax_dict_to_list(crop_resolution_dict, ["z", "y", "x"])
                        crop_shape = utils.ax_dict_to_list(crop_shape_dict, ["z", "y", "x"])

                        # Convert to EM voxel coordinates
                        em_offset = [int(co / er) for co, er in zip(crop_offset, em_resolution)]
                        em_shape = [int(cs * cr / er) for cs, cr, er in zip(crop_shape, crop_resolution, em_resolution)]

                        # Extract EM region for this crop
                        em_crop = np.array(
                            em_data[
                                em_offset[0] : em_offset[0] + em_shape[0],
                                em_offset[1] : em_offset[1] + em_shape[1],
                                em_offset[2] : em_offset[2] + em_shape[2],
                            ]
                        )

                        # Compute histogram
                        hist, bin_edges = np.histogram(em_crop.flatten(), bins=256, range=(0, 256))

                        # Write histogram to CSV
                        for intensity, count in enumerate(hist):
                            if count > 0:  # Only write non-zero counts
                                writer.writerow([dataset_name, crop_name, intensity, int(count)])

                        click.echo(f"Processed {dataset_name}/{crop_name}", err=True)

            except Exception as e:
                click.echo(f"ERROR processing {dataset_name}: {e}", err=True)


if __name__ == "__main__":
    main()
