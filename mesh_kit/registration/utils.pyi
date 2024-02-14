from typing import overload

import trimesh
from numpy import typing as npt

@overload
def normalize(
    vertices: npt.NDArray, centroid: npt.NDArray, scale: float
) -> npt.NDArray: ...
@overload
def normalize(
    mesh: trimesh.Trimesh, centroid: npt.NDArray, scale: float
) -> trimesh.Trimesh: ...
@overload
def denormalize(
    vertices: npt.NDArray, centroid: npt.NDArray, scale: float
) -> npt.NDArray: ...
@overload
def denormalize(
    mesh: trimesh.Trimesh, centroid: npt.NDArray, scale: float
) -> trimesh.Trimesh: ...
