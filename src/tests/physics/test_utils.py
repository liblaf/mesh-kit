import numpy as np
import numpy.typing as npt
import pytest
import pyvista as pv

import mkit.creation
import mkit.creation.tetmesh
import mkit.physics
import mkit.physics.utils


def hess_coords_naive(_tetra: npt.ArrayLike) -> npt.NDArray[np.integer]:
    tetra: npt.NDArray[np.integer] = np.asarray(_tetra)
    n_cells: int = len(tetra)
    coords: npt.NDArray[np.integer] = np.zeros((4, n_cells * 4 * 3 * 4 * 3), int)
    for c, u, i, v, j in np.ndindex(n_cells, 4, 3, 4, 3):
        idx: int = np.ravel_multi_index((c, u, i, v, j), (n_cells, 4, 3, 4, 3))
        coords[0, idx] = tetra[c][u]
        coords[1, idx] = i
        coords[2, idx] = tetra[c][v]
        coords[3, idx] = j
    return coords


@pytest.fixture
def tetra() -> npt.NDArray[np.integer]:
    mesh: pv.UnstructuredGrid = mkit.creation.tetmesh.box()
    tetra: npt.NDArray[np.integer] = mesh.cells_dict[pv.CellType.TETRA]
    return tetra


def test_hess_coords(tetra: npt.NDArray[np.integer]) -> None:
    actual: npt.NDArray[np.integer] = mkit.physics.utils.hess_coords(tetra)
    desired: npt.NDArray[np.integer] = hess_coords_naive(tetra)
    np.testing.assert_equal(actual, desired)
