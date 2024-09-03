import numpy as np
import pyvista as pv

import mkit


def tetrahedron() -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = mkit.io.make_tet_mesh(
        [
            [1, 0, 0],
            [-1 / 3, -np.sqrt(2 / 3), -np.sqrt(2) / 3],
            [-1 / 3, 0, 2 * np.sqrt(2) / 3],
            [-1 / 3, np.sqrt(2 / 3), -np.sqrt(2) / 3],
        ],
        [[0, 1, 2, 3]],
    )
    return mesh
