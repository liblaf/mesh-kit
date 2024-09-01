from collections.abc import Sequence

import numpy as np
import pyvista as pv

import mkit.ext
import mkit.io


def box(
    bounds: Sequence[float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
) -> pv.UnstructuredGrid:
    surface: pv.PolyData = pv.Box(bounds)
    tetmesh: pv.UnstructuredGrid = mkit.ext.tetgen(surface)
    return tetmesh


def cylinder() -> pv.UnstructuredGrid:
    surface: pv.PolyData = pv.Cylinder()
    tetmesh: pv.UnstructuredGrid = mkit.ext.tetwild(surface)
    return tetmesh


def tetrahedron() -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = mkit.io.unstructured_grid_tetmesh(
        [
            [1, 0, 0],
            [-1 / 3, -np.sqrt(2 / 3), -np.sqrt(2) / 3],
            [-1 / 3, 0, 2 * np.sqrt(2) / 3],
            [-1 / 3, np.sqrt(2 / 3), -np.sqrt(2) / 3],
        ],
        [[0, 1, 2, 3]],
    )
    return mesh
