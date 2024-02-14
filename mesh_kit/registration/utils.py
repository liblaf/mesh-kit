import numpy as np
import trimesh
from numpy import typing as npt


def normalize(
    x: npt.NDArray | trimesh.Trimesh, centroid: npt.NDArray, scale: float
) -> npt.NDArray | trimesh.Trimesh:
    match x:
        case np.ndarray():
            return (x - centroid) / scale
        case trimesh.Trimesh():
            mesh: trimesh.Trimesh = x.copy()
            mesh.apply_translation(-centroid)
            mesh.apply_scale(1.0 / scale)
            return mesh
        case _:
            raise NotImplementedError()


def denormalize(
    x: npt.NDArray | trimesh.Trimesh, centroid: npt.NDArray, scale: float
) -> npt.NDArray | trimesh.Trimesh:
    match x:
        case np.ndarray():
            return scale * x + centroid
        case trimesh.Trimesh():
            mesh: trimesh.Trimesh = x.copy()
            mesh.apply_scale(scale)
            mesh.apply_translation(centroid)
            return mesh
        case _:
            raise NotImplementedError()
