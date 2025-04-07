import argparse
import os

import dask
import fibsem_tools as fst
import neuroglancer
import neuroglancer.cli
import numpy as np


def create_lv(path, volume_type="segmentation", array_name="raw"):
    z_arr = fst.read(os.path.join(path, array_name))
    arr = dask.array.transpose(fst.io.zarr.to_dask(z_arr), (2, 3, 4, 1, 0))
    if volume_type == "segmentation":
        arr = arr.astype("uint8")

    dim_names = ["z", "y", "x", "c", "n"]
    dim_units = ["nm", "nm", "nm", "", ""]
    dim_scales = [*z_arr.attrs["resolution"], 1, 1]
    voxel_offset = [
        *(np.array(z_arr.attrs["offset"]) / np.array(z_arr.attrs["resolution"])),
        0,
        0,
    ]
    for dim in range(arr.ndim)[::-1]:
        if arr.shape[dim] == 1:
            arr = dask.array.squeeze(arr, axis=dim)
            dim_names.pop(dim)
            dim_units.pop(dim)
            dim_scales.pop(dim)
            voxel_offset.pop(dim)
    dims = neuroglancer.CoordinateSpace(
        names=dim_names, units=dim_units, scales=dim_scales
    )
    return neuroglancer.LocalVolume(
        arr, dimensions=dims, volume_type=volume_type, voxel_offset=voxel_offset
    )


def add_example_layers(state, snapshot_path, *, add_time=True):
    layers = {
        "raw": "image",
        "output": "image",
        "norm_output": "image",
        "labels": "segmentation",
        "mask": "segmentation",
    }
    if add_time:
        lv_func = create_lv_stacked
    else:
        lv_func = create_lv
    for layer_name, layer_type in layers.items():
        if layer_type == "image":
            state.layers[layer_name] = neuroglancer.ImageLayer(
                source=lv_func(
                    snapshot_path, volume_type=layer_type, array_name=layer_name
                )
            )
        else:
            state.layers[layer_name] = neuroglancer.SegmentationLayer(
                source=lv_func(
                    snapshot_path, volume_type=layer_type, array_name=layer_name
                )
            )


def create_lv_stacked(snapshot_path, volume_type="segmentation", array_name="raw"):
    snapshots = sorted(os.listdir(snapshot_path))
    dask_arrs = []
    for snapshot in snapshots:
        z_arr = fst.read(os.path.join(snapshot_path, snapshot, array_name))
        dask_arr = dask.array.transpose(fst.io.zarr.to_dask(z_arr), (2, 3, 4, 1, 0))
        if volume_type == "segmentation":
            dask_arr = dask_arr.astype("uint8")
        dask_arrs.append(dask_arr)
    dask_arrs = dask.array.stack(dask_arrs, axis=-1)
    dim_names = ["z", "y", "x", "c", "n", "t"]
    dim_units = ["nm", "nm", "nm", "", "", "s"]
    dim_scales = [*z_arr.attrs["resolution"], 1, 1, 1]
    voxel_offset = [
        *(np.array(z_arr.attrs["offset"]) / np.array(z_arr.attrs["resolution"])),
        0,
        0,
        0,
    ]
    for dim in range(dask_arrs.ndim)[::-1]:
        if dask_arrs.shape[dim] == 1:
            dask_arrs = dask.array.squeeze(dask_arrs, axis=dim)
            dim_names.pop(dim)
            dim_units.pop(dim)
            dim_scales.pop(dim)
            voxel_offset.pop(dim)
    dims = neuroglancer.CoordinateSpace(
        names=dim_names, units=dim_units, scales=dim_scales
    )
    return neuroglancer.LocalVolume(
        dask_arrs, dimensions=dims, volume_type=volume_type, voxel_offset=voxel_offset
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot_path", type=str)
    ap.add_argument("--add_time", action="store_true")
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        add_example_layers(s, args.snapshot_path, add_time=args.add_time)
    print(viewer)  # noqa: T201
    while True:
        signal = input("Enter q to exit: ")
        if "Q" in signal.upper():
            break
