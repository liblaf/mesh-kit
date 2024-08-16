import functools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.typing as jxt
import mkit.ext
import mkit.logging
import mkit.point
import mkit.sparse
import numpy as np
import numpy.typing as npt
import pyvista as pv
import rich
import rich.traceback
import scipy
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import taichi as ti
from icecream import ic
from mkit.physics.cell import tetra
from mkit.physics.energy import gravity
from mkit.physics.energy.abc import CellEnergyFn
from mkit.physics.energy.elastic import corotated
from mkit.physics.model import Model
from scipy.optimize import OptimizeResult

if TYPE_CHECKING:
    import sparse

ti.init()
mkit.logging.init()
rich.traceback.install(show_locals=True)


def linear(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike] = {},
    cell_data: Mapping[str, jxt.ArrayLike] = {},
    field_data: Mapping[str, jxt.ArrayLike] = {},
) -> jax.Array:
    _: Any
    mu: jax.Array = jnp.asarray(cell_data["mu"])
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])
    grad_op: jax.Array = tetra.grad_op(points)
    grad_u: jax.Array = grad_op @ disp
    E: jax.Array = 0.5 * (grad_u.T + grad_u)
    W: jax.Array = 0.5 * lambda_ * jnp.trace(E) ** 2 + mu * jnp.trace(E @ E)
    return W


class Problem:
    model: Model
    energy_fn: CellEnergyFn

    def __init__(self, mesh: Any) -> None:
        self.model = Model(mesh)
        self.energy_fn = CellEnergyFn(linear) + CellEnergyFn(gravity)

    def solve(self) -> OptimizeResult:
        x0: npt.NDArray[np.float64] = np.zeros((self.n_free, 3))
        res: OptimizeResult = scipy.optimize.minimize(
            self.fun,
            x0.flatten(),
            method="trust-constr",
            jac=self.jac,
            hess=self.hess,
            options={"disp": True, "verbose": 2},
            callback=self.callback,
        )
        # import cyipopt

        # A: scipy.sparse.coo_matrix = self.slide_constraint.A
        # ic(A.dtype)
        # res: OptimizeResult = cyipopt.minimize_ipopt(
        #     self.fun,
        #     x0.flatten(),
        #     jac=self.jac,
        #     hess=self.hess,
        #     constraints=[
        #         # self.slide_constraint
        #         {
        #             "type": "eq",
        #             "fun": lambda x: A @ x,
        #             "jac": lambda x: scipy.sparse.coo_array(A),
        #             "hess": lambda x, v: np.zeros((self.n_free * 3,)),
        #         }
        #     ],
        #     options={"print_level": 5, "max_wall_time": 300.0},
        # )
        return res

    @mkit.logging.log_time
    def fun(self, x: npt.ArrayLike) -> float:
        disp: npt.NDArray[np.float64] = self.make_disp(x)
        energy: jax.Array = self.model.energy(self.energy_fn, disp)
        return float(energy)

    @mkit.logging.log_time
    def jac(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        disp: npt.NDArray[np.float64] = self.make_disp(x)
        jac: jax.Array = self.model.energy_jac(self.energy_fn, disp)
        return np.asarray(jac[self.free_mask].flatten())

    @mkit.logging.log_time
    def hess(self, x: npt.ArrayLike) -> scipy.sparse.coo_matrix:
        disp: npt.NDArray[np.float64] = self.make_disp(x)
        hess: sparse.COO = self.model.energy_hess(self.energy_fn, disp)
        hess = mkit.sparse.sparse_mask(
            hess,
            (
                self.free_mask,
                np.ones((3,), np.bool),
                self.free_mask,
                np.ones((3,), np.bool),
            ),
        )
        return hess.reshape((self.n_free * 3, self.n_free * 3)).to_scipy_sparse()

    def callback(self, intermediate_result: OptimizeResult) -> None:
        # ic(intermediate_result)
        pass

    @functools.cached_property
    def n_free(self) -> int:
        return np.count_nonzero(self.free_mask)

    def make_disp(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        disp: npt.NDArray[np.float64] = np.zeros((self.model.n_points, 3))
        # disp[self.pin_mask] = self.model.point_data["pin_disp"][self.pin_mask]
        disp[self.free_mask] = np.asarray(x).reshape((self.n_free, 3))
        return disp

    @functools.cached_property
    def free_mask(self) -> npt.NDArray[np.bool]:
        return ~self.pin_mask

    @functools.cached_property
    def pin_mask(self) -> npt.NDArray[np.bool]:
        return np.asarray(self.model.point_data["pin_mask"], np.bool)


def main() -> None:
    mesh: pv.UnstructuredGrid = pv.read("data/input.vtu")
    problem = Problem(mesh)
    res: OptimizeResult = problem.solve()
    ic(res)
    disp: npt.NDArray[np.float64] = problem.make_disp(res.x)
    mesh.point_data["solution"] = disp
    mesh.cell_data["energy_density"] = problem.model.energy_density(  # pyright: ignore [reportArgumentType]
        CellEnergyFn(corotated), disp
    )

    mesh.save("data/linear.vtu")


if __name__ == "__main__":
    main()
