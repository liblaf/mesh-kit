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
    import numpy as np
    import numpy.typing as npt
    import pyvista as pv


def as_polydata(mesh: Any) -> pv.PolyData:
    import pyvista as pv

    if is_polydata(mesh):
        return mesh
    if is_trimesh(mesh):
        return pv.wrap(mesh)  # pyright: ignore [reportReturnType]
    if is_meshio(mesh):
        return pv.wrap(mesh)  # pyright: ignore [reportReturnType]
    raise UnsupportedConversionError(mesh, pv.PolyData)


def as_unstructured_grid(mesh: Any) -> pv.UnstructuredGrid:
    import pyvista as pv

    if is_unstructured_grid(mesh):
        return mesh
    raise UnsupportedConversionError(mesh, pv.UnstructuredGrid)


def unstructured_grid_tetmesh(
    points: npt.ArrayLike, tetra: npt.ArrayLike
) -> pv.UnstructuredGrid:
    import numpy as np
    import pyvista as pv

    points = np.asarray(points)
    tetra = np.asarray(tetra)
    cells: npt.NDArray[np.integer] = np.hstack(
        (np.full((len(tetra), 1), 4, dtype=tetra.dtype), tetra)
    )
    celltypes: npt.NDArray[np.integer] = np.full((len(tetra),), pv.CellType.TETRA)
    return pv.UnstructuredGrid(cells.flatten(), celltypes, points)
