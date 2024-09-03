import numpy as np
import numpy.typing as npt
import pyvista as pv

import mkit.io._typing as t


def as_polydata(mesh: t.AnyTriMesh) -> pv.PolyData:
    if t.is_polydata(mesh):
        return mesh
    if t.is_meshio(mesh):
        return pv.wrap(mesh)  # pyright: ignore [reportReturnType]
    if t.is_trimesh(mesh):
        return pv.wrap(mesh)  # pyright: ignore [reportReturnType]
    raise t.UnsupportedConversionError(mesh, pv.PolyData)


def as_unstructured_grid(mesh: t.AnyTetMesh) -> pv.UnstructuredGrid:
    if t.is_unstructured_grid(mesh):
        return mesh
    if t.is_meshio(mesh):
        return pv.wrap(mesh)  # pyright: ignore [reportReturnType]
    raise t.UnsupportedConversionError(mesh, pv.UnstructuredGrid)


def make_tet_mesh(_points: npt.ArrayLike, _tetra: npt.ArrayLike) -> pv.UnstructuredGrid:
    points: npt.NDArray[np.floating] = np.asarray(_points)
    tetra: npt.NDArray[np.integer] = np.asarray(_tetra)
    cells: npt.NDArray[np.integer] = np.hstack(
        (np.full((len(tetra), 1), 4, dtype=tetra.dtype), tetra)
    )
    celltypes: npt.NDArray[np.integer] = np.full((len(tetra),), pv.CellType.TETRA)
    return pv.UnstructuredGrid(cells.flatten(), celltypes, points)
