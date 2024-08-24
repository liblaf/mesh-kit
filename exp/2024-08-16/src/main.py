from typing import Any, no_type_check

import mkit.ext
import mkit.io
import mkit.logging
import numpy as np
import numpy.typing as npt
import pyvista as pv
import taichi as ti
from icecream import ic
from mkit.logging import log_time

ti.init()

mkit.logging.init()


@log_time
def energy_hess_coords(mesh: pv.UnstructuredGrid) -> npt.NDArray[np.integer]:
    coords: ti.ScalarField = ti.field(int, (4, 144 * mesh.n_cells))
    mesh: ti.MeshInstance = mkit.io.as_taichi(mkit.io.as_meshio(mesh), ["CV"])

    @no_type_check
    @ti.kernel
    def init_coords(mesh: ti.template(), coords: ti.template()):
        for c in mesh.cells:
            for u, i, v, j in ti.ndrange(4, 3, 4, 3):
                idx = 144 * c.id + 36 * u + 12 * i + 3 * v + j
                coords[0, idx] = c.verts[u].id
                coords[1, idx] = i
                coords[2, idx] = c.verts[v].id
                coords[3, idx] = j

    init_coords(mesh, coords)
    return coords.to_numpy()  # pyright: ignore [reportReturnType]


@log_time
def hess_coords(mesh: pv.UnstructuredGrid) -> npt.NDArray[np.integer]:
    coords = np.zeros((4, 144 * tetmesh.n_cells), int)
    tetra = tetmesh.cells_dict[pv.CellType.TETRA]
    coords[0, :] = np.repeat(tetra, 3 * 4 * 3)
    coords[1, :] = np.repeat(np.tile([0, 1, 2], tetmesh.n_cells * 4), 4 * 3)
    coords[2, :] = np.repeat(np.tile(tetra, 12), 3)
    coords[3, :] = np.tile([0, 1, 2], tetmesh.n_cells * 4 * 3 * 4)
    return coords


surface = pv.Box()
tetmesh = mkit.ext.tetwild(surface, lr=0.02)
desired = energy_hess_coords(tetmesh)
coords = hess_coords(tetmesh)

np.testing.assert_allclose(coords, desired, rtol=0, atol=0)
