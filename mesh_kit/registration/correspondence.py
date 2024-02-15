import functools
from typing import Optional

import numpy as np
import pydantic
import trimesh
from numpy import typing as npt
from scipy import spatial

from mesh_kit.common import testing
from mesh_kit.std import time as _time


class Config(pydantic.BaseModel):
    threshold: float = pydantic.Field(default=0.1, ge=0.0)
    normal: float = pydantic.Field(default=0.5, ge=0.0, le=1.0)

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
    data: npt.NDArray = np.hstack(
        ((1.0 - normal) * mesh.vertices, normal * mesh.vertex_normals)
    )
    testing.assert_shape(data.shape, (mesh.vertices.shape[0], 6))
    return data


@functools.cache
@_time.timeit
def kdtree(mesh: trimesh.Trimesh, normal: float) -> spatial.KDTree:
    return spatial.KDTree(make_data(mesh, normal))


@_time.timeit
def correspondence(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    config: Optional[Config] = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    if not config:
        config = Config()
    num_vertices: int = source_mesh.vertices.shape[0]
    tree: spatial.KDTree = kdtree(target_mesh, config.normal)
    d: npt.NDArray
    i: npt.NDArray
    d, i = tree.query(make_data(source_mesh, config.normal))
    testing.assert_shape(d.shape, (num_vertices,))
    testing.assert_shape(i.shape, (num_vertices,))
    target_positions: npt.NDArray = target_mesh.vertices[i]
    testing.assert_shape(target_positions.shape, (num_vertices, 3))
    target_normals: npt.NDArray = target_mesh.vertex_normals[i]
    testing.assert_shape(target_normals.shape, (num_vertices, 3))
    return d, target_positions, target_normals
