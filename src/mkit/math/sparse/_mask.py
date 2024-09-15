from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import sparse


def mask(array: Any, coord_masks: Sequence[npt.ArrayLike]) -> sparse.COO:
    array: sparse.COO = sparse.as_coo(array)
    coord_masks: list[npt.NDArray[np.bool]] = [
        np.asarray(mask, bool) for mask in coord_masks
    ]
    data_mask: npt.NDArray[np.bool] = np.ones((array.nnz,), bool)
    new_coords: list[npt.NDArray[np.integer]] = []
    new_shape: list[int] = []
    for coord, coord_mask in zip(array.coords, coord_masks, strict=True):
        data_mask &= coord_mask[coord]
        new_shape.append(np.count_nonzero(coord_mask))
    for coord, coord_mask in zip(array.coords, coord_masks, strict=True):
        coord_map: npt.NDArray[np.integer] = np.cumsum(coord_mask) - 1
        new_coord: npt.NDArray[np.integer] = coord_map[coord[data_mask]]
        new_coords.append(new_coord)
    return sparse.COO(np.asarray(new_coords), array.data[data_mask], shape=new_shape)
