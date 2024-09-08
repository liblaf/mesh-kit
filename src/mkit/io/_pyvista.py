from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pyvista as pv

import mkit.io._typing as t

if TYPE_CHECKING:
    import trimesh as tm

    from mkit.typing import AnyMesh, AnySurfaceMesh


def as_polydata(mesh: AnySurfaceMesh) -> pv.PolyData:
    if t.is_polydata(mesh):
        return mesh
    if t.is_meshio(mesh):
        return pv.wrap(mesh)  # pyright: ignore [reportReturnType]
    if t.is_trimesh(mesh):
        return trimesh_to_polydata(mesh)
    raise t.UnsupportedConversionError(mesh, pv.PolyData)


def as_unstructured_grid(mesh: AnyMesh) -> pv.UnstructuredGrid:
    if t.is_unstructured_grid(mesh):
        return mesh
    if t.is_meshio(mesh):
        return pv.wrap(mesh)  # pyright: ignore [reportReturnType]
    raise t.UnsupportedConversionError(mesh, pv.UnstructuredGrid)


def make_tet_mesh(_points: npt.ArrayLike, _tetra: npt.ArrayLike) -> pv.UnstructuredGrid:
    points: npt.NDArray[np.floating] = np.asarray(_points)
    tetra: npt.NDArray[np.integer] = np.asarray(_tetra)
    cells: npt.NDArray[np.integer] = np.empty((len(tetra), 5), dtype=tetra.dtype)
    cells[:, 0] = 4
    cells[:, 1:] = tetra
    celltypes: npt.NDArray[np.integer] = np.full((len(tetra),), pv.CellType.TETRA)
    return pv.UnstructuredGrid(cells.flatten(), celltypes, points)


def trimesh_to_polydata(mesh: tm.Trimesh) -> pv.PolyData:
    try:
        return pv.wrap(mesh)  # pyright: ignore [reportReturnType]
    except NotImplementedError:
        return pv.make_tri_mesh(mesh.vertices, mesh.faces)
