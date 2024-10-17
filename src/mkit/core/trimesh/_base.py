from __future__ import annotations

import pyvista as pv

import mkit
import mkit.math as m
import mkit.typing as t
import mkit.typing.numpy as tn


class TriMeshBase:
    _data: pv.PolyData

    def __init__(
        self,
        points: tn.FN3Like,
        faces: tn.IN3Like,
        point_data: t.AttrsLike | None = None,
        cell_data: t.AttrsLike | None = None,
        field_data: t.AttrsLike | None = None,
    ) -> None:
        self._data = pv.PolyData.from_regular_faces(
            m.as_numpy(points), m.as_numpy(faces)
        )
        if point_data:
            self._data.point_data.update(
                {k: m.as_numpy(v) for k, v in point_data.items()}
            )
        if cell_data:
            self._data.cell_data.update(
                {k: m.as_numpy(v) for k, v in cell_data.items()}
            )
        if field_data:
            self._data.field_data.update(
                {k: m.as_numpy(v) for k, v in field_data.items()}
            )

    @property
    def points(self) -> tn.FN3:
        return self._data.points

    @points.setter
    def points(self, points: tn.FN3Like) -> None:
        self._data.points = m.as_numpy(points)

    @property
    def faces(self) -> tn.IN3:
        return self._data.regular_faces

    @property
    def point_data(self) -> mkit.Attrs:
        return mkit.Attrs(self._data.point_data)

    @property
    def cell_data(self) -> mkit.Attrs:
        return mkit.Attrs(self._data.cell_data)

    @property
    def field_data(self) -> mkit.Attrs:
        return mkit.Attrs(self._data.field_data)
