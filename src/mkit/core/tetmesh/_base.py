from __future__ import annotations

import numpy as np
import pyvista as pv

import mkit
import mkit.math as mm
import mkit.typing as mt
import mkit.typing.numpy as tn


class TetMeshBase(mkit.DataObject[pv.UnstructuredGrid]):
    _data: pv.UnstructuredGrid

    def __init__(
        self,
        points: tn.FN3Like | None = None,
        tetras: tn.IN3Like | None = None,
        point_data: mt.AttrsLike | None = None,
        cell_data: mt.AttrsLike | None = None,
        field_data: mt.AttrsLike | None = None,
    ) -> None:
        if points is None or tetras is None:
            self._data = pv.UnstructuredGrid()
            return
        points: tn.FN3 = mm.as_numpy(points)
        tetras: tn.IN4 = mm.as_numpy(tetras)
        cells: pv.CellArray = pv.CellArray.from_regular_cells(tetras)
        cell_types: tn.IN = np.full((cells.n_cells,), pv.CellType.TETRA)
        self._data = pv.UnstructuredGrid(cells, cell_types, points)
        self.point_data = point_data
        self.cell_data = cell_data
        self.field_data = field_data

    @property
    def tetras(self) -> tn.IN4:
        return self._data.cells_dict[pv.CellType.TETRA]
