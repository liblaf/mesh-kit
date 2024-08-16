from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import sparse


def sparse_mask(array: sparse.COO, coord_masks: Sequence[npt.ArrayLike]) -> sparse.COO:
    coord_masks = tuple(np.asarray(mask, np.bool) for mask in coord_masks)
    data_mask: npt.NDArray[np.bool] = np.ones((array.nnz,), np.bool)
    coords_masked: tuple[npt.NDArray[np.bool], ...] = ()
    for coord, coord_mask in zip(array.coords, coord_masks, strict=True):
        data_mask &= coord_mask[coord]
    for coord, coord_mask in zip(array.coords, coord_masks, strict=True):
        coord_map: npt.NDArray[np.bool] = np.cumsum(coord_mask) - 1
        coord_masked: npt.NDArray[np.bool] = coord_map[coord[data_mask]]
        coords_masked += (coord_masked,)
    return sparse.COO(
        coords_masked,
        array.data[data_mask],
        shape=tuple(np.count_nonzero(mask) for mask in coord_masks),
    )
