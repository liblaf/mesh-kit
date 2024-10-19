from __future__ import annotations

import pyvista as pv

import mkit
import mkit.math as mm
import mkit.typing as mt
import mkit.typing.numpy as tn


class TriMeshBase(mkit.DataObject[pv.PolyData]):
    _data: pv.PolyData

    def __init__(
        self,
        points: tn.FN3Like | None = None,
        faces: tn.IN3Like | None = None,
        point_data: mt.AttrsLike | None = None,
        cell_data: mt.AttrsLike | None = None,
        field_data: mt.AttrsLike | None = None,
    ) -> None:
        if points is None or faces is None:
            self._data = pv.PolyData()
            return
        self._data = pv.PolyData.from_regular_faces(
            mm.as_numpy(points), mm.as_numpy(faces)
        )
        self.point_data = point_data
        self.cell_data = cell_data
        self.field_data = field_data

    @property
    def faces(self) -> tn.IN3:
        return self._data.regular_faces
