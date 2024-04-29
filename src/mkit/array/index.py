import numpy as np
from numpy import typing as npt
from scipy import spatial


def position_to_index(
    verts: npt.ArrayLike, pos: npt.ArrayLike, *, distance_upper_bound: float = np.inf
) -> npt.NDArray[np.intp]:
    """
    Args:
        verts: (V, 3) float
        pos: (N, 3) float
        distance_upper_bound: float

    Returns:
        (N,) int
    """
    verts = np.asarray(verts)
    pos = np.asarray(pos)
    kdtree = spatial.KDTree(verts)
    distance: npt.NDArray[np.float64]
    index: npt.NDArray[np.intp]
    distance, index = kdtree.query(pos, k=1, distance_upper_bound=distance_upper_bound)
    return index
