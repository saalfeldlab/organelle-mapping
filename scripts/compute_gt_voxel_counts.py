"""Compute ground truth voxel counts per (dataset, crop, label, scale) and store in a database.

Reads label metadata from zarr attributes (complement_counts) without loading arrays.
Discovers labels from zarr group metadata (class_names).
"""

import logging
from pathlib import Path

import click
import numpy as np
import yaml
import zarr
from pydantic import TypeAdapter

from organelle_mapping.config.data import DataConfig
from organelle_mapping.database import init_database, insert_crop

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.argument("data-config", type=click.Path(exists=True))
@click.option("--db-url", required=True, help="Database URL (e.g. 'sqlite:///results.db')")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
)
def main(data_config, db_url, log_level):
    """Compute GT voxel counts from label metadata and store in the crops DB table."""
    logger.setLevel(log_level.upper())

    # Load data config
    with open(data_config) as f:
        data_cfg = TypeAdapter(DataConfig).validate_python(
            yaml.safe_load(f), context={"base_dir": Path(data_config).parent}
        )

    engine = init_database(db_url)

    for ds_name, ds_info in data_cfg.datasets.items():
        logger.info(f"Processing dataset: {ds_name}")
        labels_base = f"{ds_info.labels.data}/{ds_info.labels.group}"

        # Iterate crops
        for crop_group in ds_info.labels.crops:
            for crop_raw in crop_group.split(","):
                crop_name = crop_raw.strip()
                crop_path = f"{labels_base}/{crop_name}"
                logger.info(f"  Crop: {crop_name}")

                try:
                    crop_grp = zarr.open(crop_path, mode="r")
                except Exception as e:
                    logger.warning(f"  Could not open {crop_path}: {e}")
                    continue

                # Discover labels from metadata
                try:
                    class_names = crop_grp.attrs["cellmap"]["annotation"]["class_names"]
                except KeyError:
                    logger.warning(f"  No class_names metadata in {crop_path}, skipping")
                    continue

                logger.info(f"  Labels: {class_names}")

                for label_name in class_names:
                    try:
                        label_grp = crop_grp[label_name]
                    except KeyError:
                        logger.warning(f"  Label group '{label_name}' not found in {crop_path}, skipping")
                        continue

                    # Get scale levels
                    scale_levels = sorted(label_grp.keys(), key=lambda x: int(x[1:]))

                    for scale_level in scale_levels:
                        try:
                            scale_grp = label_grp[scale_level]
                            ann_attrs = scale_grp.attrs["cellmap"]["annotation"]
                            complement_counts = ann_attrs["complement_counts"]

                            # Get voxel counts
                            total_voxels = float(np.prod(scale_grp.shape))
                            absent = float(complement_counts.get("absent", 0))
                            unknown = float(complement_counts.get("unknown", 0))
                            present = total_voxels - absent - unknown

                            # Get resolution from multiscale metadata
                            ms_attrs = label_grp.attrs.get("multiscales", [{}])[0]
                            datasets = ms_attrs.get("datasets", [])
                            axes = [ax["name"] for ax in ms_attrs.get("axes", [])]
                            resolution = {}
                            for ds in datasets:
                                if ds["path"] == scale_level:
                                    scale_values = ds["coordinateTransformations"][0]["scale"]
                                    resolution = dict(zip(axes, scale_values))
                                    break

                            if not resolution:
                                logger.warning(
                                    f"  No resolution found for {label_name}/{scale_level}, skipping"
                                )
                                continue

                            insert_crop(
                                engine,
                                dataset=ds_name,
                                crop=crop_name,
                                label=label_name,
                                scale_level=scale_level,
                                resolution_z=resolution.get("z", 0.0),
                                resolution_y=resolution.get("y", 0.0),
                                resolution_x=resolution.get("x", 0.0),
                                total_voxels=total_voxels,
                                present=present,
                                absent=absent,
                                unknown=unknown,
                            )
                            logger.debug(
                                f"    {label_name}/{scale_level}: "
                                f"present={present} absent={absent} unknown={unknown} "
                                f"total={total_voxels} (res: {resolution})"
                            )

                        except KeyError as e:
                            logger.warning(
                                f"  Missing metadata for {label_name}/{scale_level}: {e}, skipping"
                            )
                            continue

    logger.info("Done.")


if __name__ == "__main__":
    main()
