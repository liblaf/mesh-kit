from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

import numpy as np
import numpy.typing as npt
import pyvista as pv

import mkit.io._register as r
from mkit.io._typing import ClassName as C  # noqa: N814

if TYPE_CHECKING:
    from mkit.typing import AnyMesh


def as_unstructured_grid(mesh: AnyMesh) -> pv.UnstructuredGrid:
    return r.convert(mesh, pv.UnstructuredGrid)


def is_unstructured_grid(mesh: AnyMesh) -> TypeGuard[pv.UnstructuredGrid]:
    return isinstance(mesh, pv.UnstructuredGrid)


def make_tet_mesh(_points: npt.ArrayLike, _tetra: npt.ArrayLike) -> pv.UnstructuredGrid:
    points: npt.NDArray[np.floating] = np.asarray(_points)
    tetra: npt.NDArray[np.integer] = np.asarray(_tetra)
    cells: npt.NDArray[np.integer] = np.empty((len(tetra), 5), dtype=tetra.dtype)
    cells[:, 0] = 4
    cells[:, 1:] = tetra
    celltypes: npt.NDArray[np.integer] = np.full((len(tetra),), pv.CellType.TETRA)
    return pv.UnstructuredGrid(cells.flatten(), celltypes, points)


r.register(C.MESHIO, C.PYVISTA_UNSTRUCTURED_GRID)(pv.wrap)
