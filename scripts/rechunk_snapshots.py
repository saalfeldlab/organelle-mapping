#!/usr/bin/env python
"""Rechunk snapshot zarr files to have full channel chunks for neuroglancer visualization.

This script rechunks the 'output' array in snapshot zarr files IN PLACE so that the channel
dimension (d1) has chunks that span the full extent, enabling neuroglancer's c^
notation for RGB shader visualization of LSDs.

Also adds OME-NGFF 0.4 metadata with coordinateTransformations.

Usage:
    python rechunk_snapshots.py <snapshots_dir>
"""

import logging
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def add_ome_metadata(
    group: zarr.Group,
    array_name: str,
    voxel_size: tuple = (8.0, 8.0, 8.0),
    offset: tuple = (0.0, 0.0, 0.0),
    units: str = "nanometer",
):
    """Add OME-NGFF 0.4 metadata to a zarr subgroup for a specific array.

    Creates a subgroup named after the array and moves the array to 's0' within it,
    following OME-NGFF multiscale conventions.

    Args:
        group: Zarr group containing the array
        array_name: Name of the array to add metadata for
        voxel_size: Voxel size in physical units (z, y, x)
        offset: Offset/translation in physical units (z, y, x)
        units: Physical units (default: nanometer). Must be UDUNITS-2 compatible.
    """
    # Check if already in subgroup structure
    if array_name in group.group_keys():
        subgroup = group[array_name]
        if "s0" not in subgroup.array_keys():
            msg = f"Subgroup {array_name} exists but has no 's0' array"
            raise ValueError(msg)
        # Check if OME metadata already exists - if so, skip to avoid overwriting with defaults
        if "multiscales" in subgroup.attrs:
            logger.info(f"  {array_name} already has OME metadata, skipping")
            return
        logger.info(f"  {array_name} already in subgroup structure, adding metadata")
        array = subgroup["s0"]
    else:
        # Move array to subgroup/s0 structure
        logger.info(f"  Moving {array_name} to {array_name}/s0 structure")

        if array_name not in group.array_keys():
            msg = f"Array {array_name} not found in group - it may have been moved already"
            raise ValueError(msg)

        # Store reference to the array
        source_array_data = group[array_name]

        # Read all array properties before deletion
        array_shape = source_array_data.shape
        array_chunks = source_array_data.chunks
        array_dtype = source_array_data.dtype
        array_compressor = source_array_data.compressor
        array_fill_value = source_array_data.fill_value
        array_data = source_array_data[:]

        # Delete the array first to free up the name
        del group[array_name]

        # Create subgroup with the freed name
        subgroup = group.create_group(array_name)

        # Create the s0 array with the original properties
        array = subgroup.create_dataset(
            "s0",
            shape=array_shape,
            chunks=array_chunks,
            dtype=array_dtype,
            compressor=array_compressor,
            fill_value=array_fill_value,
        )

        # Copy the data
        array[:] = array_data

    ndim = len(array.shape)

    # Build axes - assuming 5D: (batch, channel, z, y, x)
    axes = []
    if ndim >= 5:  # noqa: PLR2004
        axes = [
            {"name": "b", "type": "time"},  # batch (using time type to test viewer compatibility)
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": units},
            {"name": "y", "type": "space", "unit": units},
            {"name": "x", "type": "space", "unit": units},
        ]
    elif ndim == 4:  # noqa: PLR2004
        # Assume (channel, z, y, x)
        axes = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": units},
            {"name": "y", "type": "space", "unit": units},
            {"name": "x", "type": "space", "unit": units},
        ]
    elif ndim == 3:  # noqa: PLR2004
        # Assume (z, y, x)
        axes = [
            {"name": "z", "type": "space", "unit": units},
            {"name": "y", "type": "space", "unit": units},
            {"name": "x", "type": "space", "unit": units},
        ]

    # Build coordinateTransformations
    # Scale and translation arrays must match number of dimensions
    scale = [1.0] * (ndim - 3) + list(voxel_size)  # 1.0 for batch/channel, voxel_size for spatial
    translation = [0.0] * (ndim - 3) + list(offset)  # 0.0 for batch/channel, offset for spatial

    # Update subgroup metadata with OME-NGFF structure
    subgroup.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": axes,
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": scale},
                        {"type": "translation", "translation": translation},
                    ],
                }
            ],
            "coordinateTransformations": [],  # No global transformations
            "name": array_name,
        }
    ]
    logger.info(f"  Added OME-NGFF metadata to {array_name}/s0: scale={voxel_size}, translation={offset}")


def rechunk_array_inplace(
    zarr_path: Path, array_name: str = "output", *, add_metadata: bool = True, voxel_size: tuple | None = None
):
    """Rechunk an array in a zarr file in-place and add OME-NGFF metadata.

    Args:
        zarr_path: Path to .zarr directory
        array_name: Name of array to rechunk (default: 'output')
        add_metadata: Whether to add OME-NGFF metadata (default: True)
        voxel_size: Voxel size in nm for spatial dimensions (z, y, x). If None, tries to read from array attributes.
    """
    logger.info(f"Rechunking {zarr_path}/{array_name}")

    # Open zarr group
    group = zarr.open(str(zarr_path), mode="r+")

    # Check if array exists (either as direct array or in subgroup)
    if array_name in group.array_keys():
        source_array = group[array_name]
    elif array_name in group.group_keys():
        subgroup = group[array_name]
        if "s0" in subgroup.array_keys():
            source_array = subgroup["s0"]
        else:
            logger.warning(f"Subgroup '{array_name}' exists but has no 's0' array in {zarr_path}, skipping")
            return None, None
    else:
        logger.warning(f"Array '{array_name}' not found in {zarr_path}, skipping")
        return None, None

    logger.info(f"  Current shape: {source_array.shape}, chunks: {source_array.chunks}, dtype: {source_array.dtype}")

    # Try to get voxel size from existing metadata if not provided
    if voxel_size is None:
        if "voxel_size" in source_array.attrs:
            # Read from voxel_size attribute (gunpowder convention)
            vs = source_array.attrs["voxel_size"]
            if isinstance(vs, (list, tuple)) and len(vs) >= 3:  # noqa: PLR2004
                voxel_size = tuple(float(x) for x in vs[-3:])  # Take last 3 values (z, y, x)
                logger.info(f"  Read voxel_size from array attributes: {voxel_size}")
            else:
                logger.warning(f"  'voxel_size' attribute has unexpected format: {vs}, using default (8, 8, 8)")
                voxel_size = (8.0, 8.0, 8.0)
        elif "resolution" in source_array.attrs:
            # Read from resolution attribute (alternative naming)
            resolution = source_array.attrs["resolution"]
            if isinstance(resolution, (list, tuple)) and len(resolution) >= 3:  # noqa: PLR2004
                voxel_size = tuple(float(x) for x in resolution[-3:])  # Take last 3 values (z, y, x)
                logger.info(f"  Read voxel_size from 'resolution' attribute: {voxel_size}")
            else:
                logger.warning(f"  'resolution' attribute has unexpected format: {resolution}, using default (8, 8, 8)")
                voxel_size = (8.0, 8.0, 8.0)
        else:
            logger.info("  No 'voxel_size' or 'resolution' attribute found, using default: (8, 8, 8)")
            voxel_size = (8.0, 8.0, 8.0)

    # Try to get offset from existing metadata
    offset = (0.0, 0.0, 0.0)
    if "offset" in source_array.attrs:
        offset_attr = source_array.attrs["offset"]
        if isinstance(offset_attr, (list, tuple)) and len(offset_attr) >= 3:  # noqa: PLR2004
            offset = tuple(float(x) for x in offset_attr[-3:])  # Take last 3 values (z, y, x)
            logger.info(f"  Read offset from array attributes: {offset}")

    # Determine channel dimension index based on array dimensionality
    # 4D: (channel, z, y, x) - channel is dim 0
    # 5D: (batch, channel, z, y, x) - channel is dim 1
    ndim = len(source_array.shape)
    if ndim == 4:  # noqa: PLR2004
        channel_dim = 0
    elif ndim == 5:  # noqa: PLR2004
        channel_dim = 1
    else:
        logger.warning(f"  Unexpected number of dimensions: {ndim}, expected 4 or 5")
        # Still add metadata if requested
        if add_metadata:
            add_ome_metadata(group, array_name, voxel_size=voxel_size, offset=offset)
        return

    # Check if already has full channel chunks
    if source_array.chunks[channel_dim] == source_array.shape[channel_dim]:
        logger.info("  Already has full channel chunks, skipping rechunk")
        # Still add metadata if requested
        if add_metadata:
            add_ome_metadata(group, array_name, voxel_size=voxel_size, offset=offset)
        return

    # Set new chunks: full channel dimension, keep original spatial chunks
    new_chunks = list(source_array.chunks)
    new_chunks[channel_dim] = source_array.shape[channel_dim]
    new_chunks = tuple(new_chunks)

    logger.info(f"  Target chunks: {new_chunks}")

    # Read all data
    logger.info("  Reading data...")
    data = source_array[:]

    # Store metadata
    compressor = source_array.compressor
    dtype = source_array.dtype
    shape = source_array.shape

    # Determine if we're working with subgroup structure
    in_subgroup = array_name in group.group_keys()

    # Create temporary array with new chunks
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "temp.zarr"
        logger.info("  Writing to temporary location with new chunks...")

        temp_array = zarr.open(
            str(temp_path), mode="w", shape=shape, chunks=new_chunks, dtype=dtype, compressor=compressor
        )

        # Write data
        temp_array[:] = data

        # Delete old array
        logger.info("  Deleting old array...")
        if in_subgroup:
            del group[array_name]["s0"]
            # Copy temporary array back to subgroup
            logger.info(f"  Copying rechunked array back to {array_name}/s0...")
            zarr.copy(temp_array, group[array_name], name="s0")
        else:
            del group[array_name]
            # Copy temporary array back with new chunks
            logger.info("  Copying rechunked array back...")
            zarr.copy(temp_array, group, name=array_name)

    logger.info(f"  Rechunked successfully: {new_chunks}")

    # Add OME metadata if requested (this will create subgroup structure if not already present)
    if add_metadata:
        add_ome_metadata(group, array_name, voxel_size=voxel_size, offset=offset)


def rechunk_worker(args):
    """Worker function for parallel rechunking."""
    zarr_path, array_name, add_metadata, voxel_size = args
    try:
        rechunk_array_inplace(zarr_path, array_name, add_metadata=add_metadata, voxel_size=voxel_size)
        return (zarr_path, array_name, True, None)
    except Exception as e:
        return (zarr_path, array_name, False, str(e))


@click.command()
@click.argument("snapshots_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--array", default=None, help="Name of specific array to rechunk (default: all arrays)")
@click.option("--workers", "-w", default=None, type=int, help="Number of parallel workers (default: CPU count)")
@click.option("--no-metadata", is_flag=True, help="Skip adding OME-NGFF metadata")
@click.option(
    "--voxel-size",
    default=None,
    help='Voxel size in nm (z,y,x). If not specified, reads from array "resolution" attribute.',
)
def main(
    snapshots_dir: Path,
    *,
    array: str | None = None,
    workers: int | None = None,
    no_metadata: bool = False,
    voxel_size: str | None = None,
):
    """Rechunk snapshot zarr files in SNAPSHOTS_DIR for neuroglancer visualization.

    This modifies the zarr files IN PLACE to have full channel chunks and adds OME-NGFF 0.4 metadata.
    """
    logger.info(f"Rechunking snapshots in {snapshots_dir}")

    # Parse voxel size if provided
    add_metadata = not no_metadata
    voxel_size_tuple = None
    if voxel_size is not None:
        voxel_size_tuple = tuple(float(x) for x in voxel_size.split(","))
        if len(voxel_size_tuple) != 3:  # noqa: PLR2004
            msg = f"voxel_size must have 3 values (z,y,x), got: {voxel_size}"
            raise ValueError(msg)
        logger.info(f"Using provided voxel_size={voxel_size_tuple} nm")
    else:
        logger.info("Will try to read voxel_size from array attributes")

    if not add_metadata:
        logger.info("Skipping OME-NGFF metadata (--no-metadata)")

    # Find all .zarr directories
    zarr_paths = sorted(snapshots_dir.glob("*.zarr"))

    if not zarr_paths:
        logger.warning(f"No .zarr directories found in {snapshots_dir}")
        return

    if workers is None:
        workers = cpu_count()

    logger.info(f"Found {len(zarr_paths)} snapshot(s) to rechunk using {workers} workers")

    # Prepare arguments for workers
    if array is None:
        # Process all arrays - first get list of arrays from first zarr
        # Check both direct arrays and subgroups (already processed arrays are in subgroups)
        sample_group = zarr.open(str(zarr_paths[0]), mode="r")
        array_names = list(sample_group.array_keys())
        # Also include subgroups that contain 's0' arrays (already processed)
        for group_name in sample_group.group_keys():
            if "s0" in sample_group[group_name].array_keys():
                if group_name not in array_names:
                    array_names.append(group_name)
        logger.info(f"Will process all arrays: {array_names}")
        work_items = [
            (zarr_path, array_name, add_metadata, voxel_size_tuple)
            for zarr_path in zarr_paths
            for array_name in array_names
        ]
    else:
        work_items = [(zarr_path, array, add_metadata, voxel_size_tuple) for zarr_path in zarr_paths]

    # Process in parallel
    with Pool(workers) as pool:
        results = pool.map(rechunk_worker, work_items)

    # Report results
    successes = sum(1 for _, _, success, _ in results if success)
    failures = [(path, array_name, error) for path, array_name, success, error in results if not success]

    logger.info(f"Rechunking complete! {successes}/{len(work_items)} succeeded")

    if failures:
        logger.error(f"{len(failures)} failures:")
        for path, array_name, error in failures:
            logger.error(f"  {path}/{array_name}: {error}")


if __name__ == "__main__":
    main()
