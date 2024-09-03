from collections.abc import Sequence

import pyvista as pv

import mkit


def box(
    bounds: Sequence[float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
) -> pv.UnstructuredGrid:
    surface: pv.PolyData = pv.Box(bounds)
    tetmesh: pv.UnstructuredGrid = mkit.ext.tetgen(surface)
    return tetmesh
