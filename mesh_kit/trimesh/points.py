import trimesh
from numpy import typing as npt

from mesh_kit.typing import check_shape as _check_shape


def pos2idx(mesh: trimesh.Trimesh, points: npt.NDArray) -> npt.NDArray:
    num_points: int = points.shape[0]
    points = _check_shape(points, (num_points, 3))
    distance: npt.NDArray
    vertex_id: npt.NDArray
    distance, vertex_id = mesh.nearest.vertex(points)
    distance = _check_shape(distance, (num_points,))
    return _check_shape(vertex_id, (num_points,))
