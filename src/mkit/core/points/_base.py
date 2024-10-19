from __future__ import annotations

import pyvista as pv

import mkit
import mkit.math as mm
import mkit.typing as mt
import mkit.typing.numpy as tn


class PointCloudBase(mkit.DataObject[pv.PolyData]):
    _data: pv.PolyData

    def __init__(
        self, points: tn.FN3Like | None = None, point_data: mt.AttrsLike | None = None
    ) -> None:
        if points is None:
            self._data = pv.PolyData()
            return
        self._data = pv.PolyData(mm.as_numpy(points))
        self.point_data = point_data
