from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np
import pyvista as pv

from mkit.io._register import REGISTRY
from mkit.io._typing import ClassName as C  # noqa: N814

if TYPE_CHECKING:
    import mkit.typing.numpy as tn


def as_unstructured_grid(mesh: Any) -> pv.UnstructuredGrid:
    return REGISTRY.convert(mesh, pv.UnstructuredGrid)


def is_unstructured_grid(mesh: Any) -> TypeGuard[pv.UnstructuredGrid]:
    return isinstance(mesh, pv.UnstructuredGrid)


def make_tet_mesh(points: tn.FN3Like, tetra: tn.IN4Like) -> pv.UnstructuredGrid:
    points: tn.FN3 = np.asarray(points)
    tetra: tn.IN4 = np.asarray(tetra)
    cells: pv.CellArray = pv.CellArray.from_regular_cells(tetra)
    celltypes: tn.IN = np.full((cells.n_cells,), pv.CellType.TETRA)
    return pv.UnstructuredGrid(cells, celltypes, points)


REGISTRY.register(C.MESHIO, C.PYVISTA_UNSTRUCTURED_GRID)(pv.wrap)
