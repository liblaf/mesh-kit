from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mkit.io.typing import (
    UnsupportedMeshError,
    is_meshio,
    is_polydata,
    is_taichi,
    is_trimesh,
)

if TYPE_CHECKING:
    import meshio
    import pyvista as pv
    import taichi as ti
    import trimesh


def as_meshio(mesh: Any) -> meshio.Mesh:
    if is_meshio(mesh):
        return mesh
    if is_polydata(mesh):
        return polydata_to_meshio(mesh)
    if is_taichi(mesh):
        return taichi_to_meshio(mesh)
    if is_trimesh(mesh):
        return trimesh_to_meshio(mesh)
    raise UnsupportedMeshError(mesh)


def polydata_to_meshio(mesh: pv.PolyData) -> meshio.Mesh:
    import meshio

    return meshio.Mesh(points=mesh.points, cells=[("triangle", mesh.regular_faces)])


def taichi_to_meshio(mesh: ti.MeshInstance) -> meshio.Mesh:
    raise NotImplementedError


def trimesh_to_meshio(mesh: trimesh.Trimesh) -> meshio.Mesh:
    import meshio

    return meshio.Mesh(points=mesh.vertices, cells=[("triangle", mesh.faces)])
