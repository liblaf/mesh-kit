from typing import Any

import numpy as np
import numpy.typing as npt
import pyvista as pv
from icecream import ic

from mkit.io.typing import (
    UnsupportedMeshError,
    is_meshio,
    is_polydata,
    is_trimesh,
    is_unstructured_grid,
)


def as_polydata(mesh: Any) -> pv.PolyData:
    if is_polydata(mesh):
        return mesh
    if is_trimesh(mesh):
        return pv.wrap(mesh)  # pyright: ignore [reportReturnType]
    if is_meshio(mesh):
        return pv.wrap(mesh)  # pyright: ignore [reportReturnType]
    raise UnsupportedMeshError(mesh)


def as_unstructured_grid(mesh: Any) -> pv.UnstructuredGrid:
    if is_unstructured_grid(mesh):
        return mesh
    raise UnsupportedMeshError(mesh)


def unstructured_grid_tetmesh(
    points: npt.ArrayLike, tetra: npt.ArrayLike
) -> pv.UnstructuredGrid:
    points = np.asarray(points)
    tetra = np.asarray(tetra)
    cells: npt.NDArray[np.integer] = np.hstack(
        (np.full((len(tetra), 1), 4, dtype=tetra.dtype), tetra)
    )
    celltypes: npt.NDArray[np.integer] = np.full((len(tetra),), pv.CellType.TETRA)
    return pv.UnstructuredGrid(cells.flatten(), celltypes, points)
