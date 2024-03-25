from typing import Any

import scipy.spatial
from numpy import typing as npt


def pos2idx(points: npt.NDArray, pos: npt.NDArray) -> npt.NDArray:
    _: Any
    tree = scipy.spatial.KDTree(points)
    idx: npt.NDArray
    _, idx = tree.query(pos)
    return idx
