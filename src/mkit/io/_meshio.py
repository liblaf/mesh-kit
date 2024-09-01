from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mkit.io.typing import (
    UnsupportedConversionError,
    is_meshio,
    is_polydata,
    is_trimesh,
    is_unstructured_grid,
)

if TYPE_CHECKING:
    import meshio
    import pyvista as pv
    import trimesh


def as_meshio(mesh: Any) -> meshio.Mesh:
    import meshio

    if is_meshio(mesh):
        return mesh
    if is_polydata(mesh):
        return polydata_to_meshio(mesh)
    if is_trimesh(mesh):
        return trimesh_to_meshio(mesh)
    if is_unstructured_grid(mesh):
        return unstructured_grid_to_meshio(mesh)
    raise UnsupportedConversionError(mesh, meshio.Mesh)


def polydata_to_meshio(mesh: pv.PolyData) -> meshio.Mesh:
    import meshio

    mesh = mesh.triangulate(progress_bar=True)
    return meshio.Mesh(points=mesh.points, cells=[("triangle", mesh.regular_faces)])


def trimesh_to_meshio(mesh: trimesh.Trimesh) -> meshio.Mesh:
    import meshio

    return meshio.Mesh(points=mesh.vertices, cells=[("triangle", mesh.faces)])


def unstructured_grid_to_meshio(mesh: pv.UnstructuredGrid) -> meshio.Mesh:
    import meshio
    import pyvista as pv

    return meshio.Mesh(
        points=mesh.points,
        cells=[("tetra", mesh.cells_dict[pv.CellType.TETRA])],
        point_data=dict(mesh.point_data),
        cell_data={k: [v] for k, v in mesh.cell_data.items()},
        field_data=dict(mesh.field_data),
    )
