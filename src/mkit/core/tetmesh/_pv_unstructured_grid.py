from __future__ import annotations

from typing import TYPE_CHECKING

from mkit.core.tetmesh._base import TetMeshBase

if TYPE_CHECKING:
    import pyvista as pv


class PyvistaUnstructuredGridMixin(TetMeshBase):
    @property
    def pyvista(self) -> pv.UnstructuredGrid:
        return self._data
