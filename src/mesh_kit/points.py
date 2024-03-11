from typing import Any

from numpy import typing as npt
from scipy import spatial


def position2idx(points: npt.NDArray, positions: npt.NDArray) -> npt.NDArray:
    _: Any
    tree = spatial.KDTree(points)
    idx: npt.NDArray
    _, idx = tree.query(positions)
    return idx
