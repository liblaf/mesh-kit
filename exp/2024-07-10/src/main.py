import dataclasses
import functools
from typing import TYPE_CHECKING

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
import scipy.optimize
import scipy.sparse
import sparse
import taichi as ti
from icecream import ic
from mkit.physics.energy import gravity
from mkit.physics.energy.abc import CellEnergyFn
from mkit.physics.energy.elastic import corotated
from mkit.physics.model import Model
from omegaconf import OmegaConf
from scipy.optimize import LinearConstraint, OptimizeResult

if TYPE_CHECKING:
    import jax

ti.init()
mkit.logging.init()
rich.traceback.install(show_locals=True)


@dataclasses.dataclass(kw_only=True)
class Config:
    mu: float = 1e3
    lambda_: float = 3e3


cfg: Config = OmegaConf.structured(Config)
cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
ic(cfg)

inner: pv.PolyData = pv.Icosphere(radius=0.1)
inner.point_data["pin_mask"] = np.ones((inner.n_points,), bool)
inner.point_data["slide_mask"] = inner.points[:, 1] > 0
# inner.point_data["slide_mask"] = np.zeros((inner.n_points,), bool)
inner.point_data["pin_mask"] &= ~inner.point_data["slide_mask"]
inner.point_data["normal"] = inner.point_normals

outer: pv.PolyData = pv.Icosphere(radius=0.2)
outer.point_data["pin_mask"] = np.zeros((outer.n_points,), bool)
outer.point_data["slide_mask"] = np.zeros((outer.n_points,), bool)
outer.point_data["normal"] = outer.point_normals

mesh: pv.UnstructuredGrid = mkit.ext.tetwild_wrapped(outer, inner)
ic(mesh)
mesh.cell_data["mu"] = np.full((mesh.n_cells,), cfg.mu)
mesh.cell_data["lambda"] = np.full((mesh.n_cells,), cfg.lambda_)
mesh.cell_data["density"] = np.full((mesh.n_cells,), 1e3)
mesh.cell_data["gravity"] = np.tile(np.asarray([0, -9.8, 0]), (mesh.n_cells, 1))


class Problem:
    model: Model
    energy_fn: CellEnergyFn

    def __init__(self) -> None:
        self.model = Model(mesh)
        self.energy_fn = CellEnergyFn(corotated) + CellEnergyFn(gravity)

        point_normal: npt.NDArray[np.float64] = np.asarray(
            self.model.point_data["normal"]
        )
        A: sparse.COO = sparse.COO(
            (np.repeat(np.arange(self.n_free), 3), np.arange(self.n_free * 3)),
            point_normal[self.free_mask].flatten(),
        )
        A = A[self.slide_mask[self.free_mask], :]
        self.slide_constraint = LinearConstraint(
            A.reshape((self.n_slide, self.n_free * 3)).to_scipy_sparse(), lb=0, ub=0
        )

    def solve(self) -> OptimizeResult:
        x0: npt.NDArray[np.float64] = np.zeros((self.n_free, 3))
        res: OptimizeResult = scipy.optimize.minimize(
            self.fun,
            x0.flatten(),
            method="trust-constr",
            jac=self.jac,
            hess=self.hess,
            constraints=[self.slide_constraint],
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

    @functools.cached_property
    def n_slide(self) -> int:
        return np.count_nonzero(self.slide_mask)

    def make_disp(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        disp: npt.NDArray[np.float64] = np.zeros((self.model.n_points, 3))
        disp[self.free_mask] = np.asarray(x).reshape((self.n_free, 3))
        return disp

    @functools.cached_property
    def free_mask(self) -> npt.NDArray[np.bool]:
        return ~self.pin_mask

    @functools.cached_property
    def pin_mask(self) -> npt.NDArray[np.bool]:
        return np.asarray(self.model.point_data["pin_mask"], np.bool)

    @functools.cached_property
    def slide_mask(self) -> npt.NDArray[np.bool]:
        return np.asarray(self.model.point_data["slide_mask"], np.bool) & ~self.pin_mask


problem = Problem()
res: OptimizeResult = problem.solve()
ic(res)
disp: npt.NDArray[np.float64] = problem.make_disp(res.x)
mesh.point_data["solution"] = disp
mesh.cell_data["energy_density"] = problem.model.energy_density(  # pyright: ignore [reportArgumentType]
    CellEnergyFn(corotated), disp
)

mesh.save("solution.vtu")
