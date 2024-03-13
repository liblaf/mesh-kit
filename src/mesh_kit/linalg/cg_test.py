from typing import no_type_check

import numpy as np
import taichi as ti
from numpy import typing as npt

from mesh_kit.linalg import cg


def test_MatrixFreeCG() -> None:
    rng: np.random.Generator = np.random.default_rng()
    A: ti.MatrixField = ti.Matrix.field(3, 3, float, shape=(16, 16))
    A_np: npt.NDArray = rng.random((16 * 3, 16 * 3))
    np.fill_diagonal(A_np, np.abs(A_np).sum(axis=1))
    A_np = A_np.reshape((16, 3, 16, 3))
    A_np = A_np.transpose((0, 2, 1, 3))
    A.from_numpy(A_np)
    x: ti.MatrixField = ti.Vector.field(3, float, shape=(16,))
    x_np: npt.NDArray = rng.random((16, 3))
    x.from_numpy(x_np)
    b: ti.MatrixField = ti.Vector.field(3, float, shape=(16,))

    @no_type_check
    @ti.kernel
    def op(x: ti.template(), Ax: ti.template()):
        for i in Ax:
            Ax[i] = 0
        for i, j in A:
            Ax[i] += A[i, j] @ x[j]

    op(x, b)
    x.fill(0)
    succeeded: bool = cg.MatrixFreeCG(
        ti.linalg.LinearOperator(op), b, x, tol=1e-6, quiet=False
    )
    assert succeeded
    np.testing.assert_allclose(x.to_numpy(), x_np, atol=1e-6)
