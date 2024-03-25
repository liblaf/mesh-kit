from typing import Any

import trimesh
from numpy import typing as npt


def inner_point(mesh: trimesh.Trimesh) -> npt.NDArray:
    _: Any
    ray: (
        trimesh.ray.ray_pyembree.RayMeshIntersector
        | trimesh.ray.ray_triangle.RayMeshIntersector
    ) = mesh.ray
    idx: int = 0
    origin: npt.NDArray = mesh.vertices[idx]
    direction: npt.NDArray = -mesh.vertex_normals[idx]
    locations: npt.NDArray
    locations, _, _ = ray.intersects_location(
        ray_origins=[origin], ray_directions=[direction]
    )
    point: npt.NDArray = (origin + locations[0]) / 2
    assert mesh.contains([point])[0]
    return point
