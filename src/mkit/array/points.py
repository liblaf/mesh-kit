import numpy as np
from numpy import typing as npt
from scipy import spatial


def position_to_index(
    points: npt.ArrayLike, pos: npt.ArrayLike, *, distance_upper_bound: float = np.inf
) -> npt.NDArray[np.integer]:
    """
    Args:
        points: (V, 3) float
        pos: (N, 3) float
        distance_upper_bound: float

    Returns:
        (N,) int
    """
    points = np.asarray(points)
    pos = np.asarray(pos)
    kdtree = spatial.KDTree(points)
    distance: npt.NDArray[np.floating]
    index: npt.NDArray[np.integer]
    distance, index = kdtree.query(pos, distance_upper_bound=distance_upper_bound)
    return index


def position_to_mask(
    points: npt.ArrayLike, pos: npt.ArrayLike, *, distance_upper_bound: float = np.inf
) -> npt.NDArray[np.bool_]:
    """
    Args:
        points: (V, 3) float
        pos: (N, 3) float
        distance_upper_bound: float

    Returns:
        (V,) bool
    """
    points = np.asarray(points)
    mask: npt.NDArray[np.bool_] = np.zeros((len(points),), np.bool_)
    idx: npt.NDArray[np.intp] = position_to_index(
        points, pos, distance_upper_bound=distance_upper_bound
    )
    mask[idx] = True
    return mask
