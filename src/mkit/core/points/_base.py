from __future__ import annotations

import pyvista as pv

import mkit
import mkit.math as m
import mkit.typing as t
import mkit.typing.numpy as tn


class PointCloudBase:
    _data: pv.PolyData

    def __init__(
        self, points: tn.FN3Like, point_data: t.AttrsLike | None = None
    ) -> None:
        self._data = pv.PolyData(m.as_numpy(points))
        if point_data:
            self._data.point_data.update(
                {k: m.as_numpy(v) for k, v in point_data.items()}
            )

    @property
    def points(self) -> tn.FN3:
        return self._data.points

    @points.setter
    def points(self, points: tn.FN3Like) -> None:
        self._data.points = m.as_numpy(points)

    @property
    def point_data(self) -> mkit.Attrs:
        return mkit.Attrs(self._data.point_data)
