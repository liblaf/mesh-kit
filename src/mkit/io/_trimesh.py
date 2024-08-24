from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mkit.io.typing import (
    UnsupportedConversionError,
    is_meshio,
    is_polydata,
    is_trimesh,
)

if TYPE_CHECKING:
    import meshio
    import pyvista as pv
    import trimesh


def as_trimesh(mesh: Any) -> trimesh.Trimesh:
    import trimesh

    if is_trimesh(mesh):
        return mesh
    if is_meshio(mesh):
        return meshio_to_trimesh(mesh)
    if is_polydata(mesh):
        return polydata_to_trimesh(mesh)
    raise UnsupportedConversionError(mesh, trimesh.Trimesh)


def meshio_to_trimesh(mesh: meshio.Mesh) -> trimesh.Trimesh:
    import trimesh

    return trimesh.Trimesh(vertices=mesh.points, faces=mesh.get_cells_type("triangle"))


def polydata_to_trimesh(mesh: pv.PolyData) -> trimesh.Trimesh:
    import trimesh

    return trimesh.Trimesh(vertices=mesh.points, faces=mesh.regular_faces)
