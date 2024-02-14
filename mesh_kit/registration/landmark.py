import trimesh
from numpy import typing as npt

from mesh_kit.common import testing


def position2index(mesh: trimesh.Trimesh, positions: npt.NDArray) -> None:
    testing.assert_shape(positions.shape, (-1, 3))
    vertex_id: npt.NDArray
    _, vertex_id = mesh.nearest.vertex(positions)
    testing.assert_shape(vertex_id.shape, (positions.shape[0],))
    return vertex_id
