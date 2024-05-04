import numpy as np
from numpy import typing as npt
from scipy import spatial


def position_to_index(
    points: npt.ArrayLike, pos: npt.ArrayLike, *, distance_upper_bound: float = np.inf
) -> npt.NDArray[np.intp]:
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
    distance: npt.NDArray[np.float64]
    index: npt.NDArray[np.intp]
    distance, index = kdtree.query(pos, distance_upper_bound=distance_upper_bound)
    return index
