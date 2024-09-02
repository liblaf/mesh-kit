from collections.abc import Sequence

import pyvista as pv

from mkit.ext import tetgen


def box(
    bounds: Sequence[float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
) -> pv.UnstructuredGrid:
    surface: pv.PolyData = pv.Box(bounds)
    tetmesh: pv.UnstructuredGrid = tetgen(surface)
    return tetmesh
