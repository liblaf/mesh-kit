from __future__ import annotations

import pyvista as pv

import mkit
import mkit.math as m
import mkit.typing as t
import mkit.typing.numpy as tn


class TriMeshBase(mkit.DataObject[pv.PolyData]):
    _data: pv.PolyData

    def __init__(
        self,
        points: tn.FN3Like | None = None,
        faces: tn.IN3Like | None = None,
        point_data: t.AttrsLike | None = None,
        cell_data: t.AttrsLike | None = None,
        field_data: t.AttrsLike | None = None,
    ) -> None:
        if points is None or faces is None:
            self._data = pv.PolyData()
            return
        self._data = pv.PolyData.from_regular_faces(
            m.as_numpy(points), m.as_numpy(faces)
        )
        self.point_data = point_data
        self.cell_data = cell_data
        self.field_data = field_data

    @property
    def faces(self) -> tn.IN3:
        return self._data.regular_faces
