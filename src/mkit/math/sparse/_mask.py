from collections.abc import Sequence
from typing import Any

import numpy as np
import sparse

import mkit
import mkit.typing.numpy as tn


def mask(array: Any, coord_masks: Sequence[tn.BNLike]) -> sparse.COO:
    array: sparse.COO = sparse.as_coo(array)
    coord_masks: list[tn.BN] = [mkit.math.numpy.as_bool(mask) for mask in coord_masks]
    data_mask: tn.BN = np.ones((array.nnz,), bool)
    new_coords: list[tn.IN] = []
    new_shape: list[int] = []
    for coord, coord_mask in zip(array.coords, coord_masks, strict=True):
        data_mask &= coord_mask[coord]
        new_shape.append(np.count_nonzero(coord_mask))
    for coord, coord_mask in zip(array.coords, coord_masks, strict=True):
        coord_map: tn.IN = np.cumsum(coord_mask) - 1
        new_coord: tn.IN = coord_map[coord[data_mask]]
        new_coords.append(new_coord)
    return sparse.COO(np.asarray(new_coords), array.data[data_mask], shape=new_shape)
