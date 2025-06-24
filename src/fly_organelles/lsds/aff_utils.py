import numpy as np
from collections.abc import Sequence
import itertools
from typing import Callable

def equality_no_bg_func(x, y):
    return (x == y) & (x > 0) & (y > 0)

def compute_affs(
    arr: np.ndarray,
    offset: Sequence[int],
    pad: bool = False,
) -> np.ndarray:
    """
    Compute affinities on a given array `arr` using the specified `offset` and distance
    function `dist_func`. if `pad` is True, `arr` will be padded s.t. the output shape
    matches the input shape.
    """
    offset = np.array(offset)
    offset_dim = len(offset)

    if pad:
        padding = []
        for axis_offset in list(offset):
            if axis_offset > 0:
                padding.append((0, axis_offset))
            else:
                padding.append((-axis_offset, 0))
        arr = np.pad(arr, pad_width=padding, mode="constant", constant_values=0)

    arr_shape = arr.shape[-offset_dim:]
    slice_ops_lower = tuple(
        slice(
            max(0, -offset[h]),
            min(arr_shape[h], arr_shape[h] - offset[h]),
        )
        for h in range(0, offset_dim)
    )
    slice_ops_upper = tuple(
        slice(
            max(0, offset[h]),
            min(arr_shape[h], arr_shape[h] + offset[h]),
        )
        for h in range(0, offset_dim)
    )

    # handle arbitrary number of leading dimensions (can be batch, channel, etc.)
    while len(slice_ops_lower) < len(arr.shape):
        slice_ops_lower = (slice(None), *slice_ops_lower)
        slice_ops_upper = (slice(None), *slice_ops_upper)

    return equality_no_bg_func(
        arr[slice_ops_lower],
        arr[slice_ops_upper],
    )

class Affs:
    def __init__(
        self,
        neighborhood: Sequence[Sequence[int]],
    ):
        self.neighborhood = neighborhood
        self.ndim = len(neighborhood[0])
        assert all(len(offset) == self.ndim for offset in neighborhood), (
            "All offsets in the neighborhood must have the same dimensionality."
        )

    def __call__(self, x):
        return np.array([
            compute_affs(x, offset, pad=True)
            for offset in self.neighborhood
        ]).astype(np.uint8)
