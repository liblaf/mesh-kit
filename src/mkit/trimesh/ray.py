from typing import Any

import trimesh
from numpy import typing as npt


def inner_point(mesh: trimesh.Trimesh) -> npt.NDArray:
    _: Any
    ray: (
        trimesh.ray.ray_pyembree.RayMeshIntersector
        | trimesh.ray.ray_triangle.RayMeshIntersector
    ) = mesh.ray
    origin: npt.NDArray = mesh.bounds[0]
    idx: int = 1
    direction: npt.NDArray = mesh.vertices[idx] - origin
    locations: npt.NDArray
    locations, _, _ = ray.intersects_location(
        ray_origins=[origin], ray_directions=[direction]
    )
    point: npt.NDArray = (locations[0] + locations[1]) / 2
    assert mesh.contains([point])[0]
    return point
