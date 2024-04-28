import numpy as np
from numpy import typing as npt
from scipy import spatial


def position2index(
    verts: npt.NDArray[np.float64],
    pos: npt.NDArray[np.float64],
    *,
    distance_upper_bound: float = np.inf,
) -> npt.NDArray[np.intp]:
    tree = spatial.KDTree(verts)
    distance: npt.NDArray[np.float64]
    index: npt.NDArray[np.intp]
    distance, index = tree.query(pos, k=1, distance_upper_bound=distance_upper_bound)
    return index
