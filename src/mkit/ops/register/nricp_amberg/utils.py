from typing import overload

import numpy as np
import trimesh
from numpy import typing as npt

from mkit._typing import check_shape as _check_shape


@overload
def normalize(
    x: trimesh.Trimesh, *, centroid: npt.NDArray, scale: float
) -> trimesh.Trimesh: ...


@overload
def normalize(
    x: npt.NDArray, *, centroid: npt.NDArray, scale: float
) -> npt.NDArray: ...


def normalize(
    x: trimesh.Trimesh | npt.NDArray, *, centroid: npt.NDArray, scale: float
) -> trimesh.Trimesh | npt.NDArray:
    match x:
        case trimesh.Trimesh():
            mesh: trimesh.Trimesh = x.copy()
            mesh.apply_translation(-centroid)
            mesh.apply_scale(1.0 / scale)
            return mesh
        case np.ndarray():
            points: npt.NDArray = _check_shape(x, (-1, 3))
            points -= centroid
            points /= scale
            return points
        case _:
            raise TypeError


@overload
def denormalize(
    x: trimesh.Trimesh, *, centroid: npt.NDArray, scale: float
) -> trimesh.Trimesh: ...


@overload
def denormalize(
    x: npt.NDArray, *, centroid: npt.NDArray, scale: float
) -> npt.NDArray: ...


def denormalize(
    x: trimesh.Trimesh | npt.NDArray, *, centroid: npt.NDArray, scale: float
) -> trimesh.Trimesh | npt.NDArray:
    match x:
        case trimesh.Trimesh():
            mesh: trimesh.Trimesh = x.copy()
            mesh.apply_scale(scale)
            mesh.apply_translation(centroid)
            return mesh
        case np.ndarray():
            points: npt.NDArray = _check_shape(x, (-1, 3))
            points *= scale
            points += centroid
            return points
        case _:
            raise TypeError
