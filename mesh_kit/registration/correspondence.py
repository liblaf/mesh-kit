import trimesh
from numpy import typing as npt
from scipy import spatial

from mesh_kit.common import testing
from mesh_kit.std import time as _time


@_time.timeit
def correspondence(
    source_mesh: trimesh.Trimesh, target_mesh: trimesh.Trimesh
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    num_vertices: int = source_mesh.vertices.shape[0]
    tree: spatial.KDTree = target_mesh.kdtree
    d: npt.NDArray
    i: npt.NDArray
    d, i = tree.query(source_mesh.vertices)
    testing.assert_shape(d.shape, (num_vertices,))
    testing.assert_shape(i.shape, (num_vertices,))
    target_positions: npt.NDArray = target_mesh.vertices[i]
    testing.assert_shape(target_positions.shape, (num_vertices, 3))
    target_normals: npt.NDArray = target_mesh.vertex_normals[i]
    testing.assert_shape(target_normals.shape, (num_vertices, 3))
    return d, target_positions, target_normals
