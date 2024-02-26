import functools

import numpy as np
import pydantic
import trimesh
from numpy import typing as npt
from scipy import spatial

from mesh_kit import log as _log
from mesh_kit.typing import check_shape as _check_shape


class Config(pydantic.BaseModel):
    threshold: pydantic.PositiveFloat = 0.1
    normal: pydantic.NonNegativeFloat = 0.5

    def __add__(self, other: "Config") -> "Config":
        return Config(
            threshold=self.threshold + other.threshold,
            normal=self.normal + other.normal,
        )

    def __sub__(self, other: "Config") -> "Config":
        return Config(
            threshold=self.threshold - other.threshold,
            normal=self.normal - other.normal,
        )

    def __rmul__(self, other: int | float) -> "Config":
        return Config(
            threshold=other * self.threshold,
            normal=other * self.normal,
        )

    def __truediv__(self, other: int | float) -> "Config":
        return (1.0 / other) * self


def make_data(mesh: trimesh.Trimesh, normal: float) -> npt.NDArray:
    data: npt.NDArray = _check_shape(
        np.hstack(((1.0 - normal) * mesh.vertices, normal * mesh.vertex_normals)),
        (mesh.vertices.shape[0], 6),
    )
    return data


@functools.cache
@_log.timeit
def kdtree(mesh: trimesh.Trimesh, normal: float) -> spatial.KDTree:
    tree: spatial.KDTree = spatial.KDTree(make_data(mesh, normal))
    return tree


@_log.timeit
def nearest(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    config: Config | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    if not config:
        config = Config()
    num_vertices: int = source_mesh.vertices.shape[0]
    tree: spatial.KDTree = kdtree(target_mesh, config.normal)
    d: npt.NDArray
    i: npt.NDArray
    d, i = tree.query(make_data(source_mesh, config.normal))
    d = _check_shape(d, (num_vertices,))
    i = _check_shape(i, (num_vertices,))
    target_positions: npt.NDArray = _check_shape(
        target_mesh.vertices[i], (num_vertices, 3)
    )
    target_normals: npt.NDArray = _check_shape(
        target_mesh.vertex_normals[i], (num_vertices, 3)
    )
    return d, target_positions, target_normals
