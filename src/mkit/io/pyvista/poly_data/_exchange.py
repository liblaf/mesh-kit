from __future__ import annotations

from typing import Any

import pyvista as pv

import mkit.io.exchange as mie
import mkit.typing as mt


def as_poly_data(
    obj: Any,
    *,
    point_data: mt.AttrsLike | None = None,
    cell_data: mt.AttrsLike | None = None,
    field_data: mt.AttrsLike | None = None,
) -> pv.PolyData:
    return mie.convert(
        obj,
        pv.PolyData,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
    )
