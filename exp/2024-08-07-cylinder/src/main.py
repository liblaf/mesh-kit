import functools
import logging
from typing import TYPE_CHECKING, Any

import jax
import mkit.logging
import mkit.sparse
import numpy as np
import numpy.typing as npt
import pyvista as pv
import scipy.optimize
import scipy.sparse
from loguru import logger
from mkit.physics.energy.abc import CellEnergy
from mkit.physics.model import Model
from mkit.physics.preset.elastic import MODELS, Config

if TYPE_CHECKING:
    import sparse


class Problem:
    model: Model
    energy_fn: CellEnergy

    def __init__(self, mesh: Any, name: str) -> None:
        self.model = Model(mesh)
        config: Config = MODELS[name]
        self.energy_fn = config.energy_fn
        for k, v in config.params.items():
            self.mesh.cell_data[k] = v  # pyright: ignore [reportArgumentType]

    def solve(self) -> scipy.optimize.OptimizeResult:
        x0: npt.NDArray[np.floating] = np.zeros((self.n_free, 3))
        res: scipy.optimize.OptimizeResult = scipy.optimize.minimize(
            self.fun,
            x0.flatten(),
            method="trust-constr",
            jac=self.jac,
            hess=self.hess,
            options={"disp": True},
        )
        return res

    def fun(self, x: npt.ArrayLike) -> jax.Array:
        disp: npt.NDArray[np.floating] = self.make_disp(x)
        energy: jax.Array = self.model.energy(self.energy_fn, disp)
        return energy

    def jac(self, x: npt.ArrayLike) -> jax.Array:
        disp: npt.NDArray[np.floating] = self.make_disp(x)
        jac: jax.Array = self.model.energy_jac(self.energy_fn, disp)
        return jac[self.free_mask].flatten()

    def hess(self, x: npt.NDArray) -> scipy.sparse.coo_matrix:
        disp: npt.NDArray[np.floating] = self.make_disp(x)
        hess: sparse.COO = self.model.energy_hess(self.energy_fn, disp)
        coord_mask: npt.NDArray[np.bool] = np.ones((3,), bool)
        hess = mkit.sparse.sparse_mask(
            hess, (self.free_mask, coord_mask, self.free_mask, coord_mask)
        )
        return hess.reshape((self.n_free * 3, self.n_free * 3)).to_scipy_sparse()

    def make_disp(self, _x: npt.ArrayLike) -> npt.NDArray[np.floating]:
        x: npt.NDArray[np.floating] = np.asarray(_x)
        disp: npt.NDArray[np.floating] = self.pin_disp.copy()
        disp[self.free_mask] = x.reshape((self.n_free, 3))
        return disp

    def energy_density(self, x: npt.ArrayLike) -> jax.Array:
        disp: npt.NDArray[np.floating] = self.make_disp(x)
        W: jax.Array = self.model.energy_density(self.energy_fn, disp)
        return W

    @property
    def mesh(self) -> pv.UnstructuredGrid:
        return self.model.mesh

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    @functools.cached_property
    def n_free(self) -> int:
        return np.count_nonzero(self.free_mask)

    @property
    def point_data(self) -> pv.DataSetAttributes:
        return self.mesh.point_data

    @functools.cached_property
    def free_mask(self) -> npt.NDArray[np.bool]:
        return ~self.pin_mask

    @functools.cached_property
    def pin_disp(self) -> npt.NDArray[np.floating]:
        pin_disp: npt.NDArray[np.floating] | None = self.point_data.get("pin_disp")
        if pin_disp is None:
            pin_disp = np.zeros((self.n_points, 3))
        return pin_disp

    @functools.cached_property
    def pin_mask(self) -> npt.NDArray[np.bool]:
        pin_mask: npt.NDArray[np.bool] | None = self.point_data.get("pin_mask")
        if pin_mask is None:
            pin_mask = np.zeros((self.n_points,), bool)
        return pin_mask


def main() -> None:
    mkit.logging.init(logging.INFO)
    mesh: pv.UnstructuredGrid = pv.read("data/input.vtu")
    for name, cfg in MODELS.items():
        logger.info("Solving {}...", cfg.name)
        problem: Problem = Problem(mesh, name)
        res: scipy.optimize.OptimizeResult = problem.solve()
        disp: npt.NDArray[np.floating] = problem.make_disp(res.x)
        mesh.point_data["solution"] = disp
        mesh.cell_data["energy_density"] = np.asarray(problem.energy_density(res.x))
        mesh.save(f"data/{name}.vtu")


if __name__ == "__main__":
    main()
