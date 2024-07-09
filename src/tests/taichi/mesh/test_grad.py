from typing import TYPE_CHECKING

import mkit.ext
import mkit.io
import mkit.taichi.mesh
import numpy as np
import taichi as ti
import trimesh

if TYPE_CHECKING:
    import meshio
    import scipy.sparse
    from numpy.typing import NDArray


def test_grad() -> None:
    ti.init(default_fp=ti.float64)
    surface: trimesh.Trimesh = trimesh.creation.icosphere()
    tetmesh: meshio.Mesh = mkit.ext.tetwild(surface)
    mesh_ti: ti.MeshInstance = mkit.io.as_taichi(tetmesh, ["CV"])
    grad: scipy.sparse.coo_matrix = mkit.taichi.mesh.grad(mesh_ti)
    deformation_gradient: NDArray[np.float64] = (grad @ tetmesh.points).reshape(
        (len(mesh_ti.cells), 3, 3)
    )
    deformation_gradient_desired: NDArray[np.float64] = np.tile(
        np.eye(3), (len(tetmesh.get_cells_type("tetra")), 1, 1)
    )
    np.testing.assert_allclose(
        deformation_gradient, deformation_gradient_desired, rtol=1e-6, atol=1e-6
    )
