from typing import TYPE_CHECKING

import jax
import numpy as np
import numpy.typing as npt
import pyvista as pv
import scipy.optimize
import scipy.sparse

import mkit
from mkit.physics import Model
from mkit.physics.energy import CellEnergy, EnergyFn
from mkit.typing import AnyTetMesh

if TYPE_CHECKING:
    import sparse


class Problem:
    model: Model
    pin_mask: npt.NDArray[np.bool]
    pin_disp: npt.NDArray[np.floating]

    def __init__(
        self,
        mesh: AnyTetMesh,
        energy_fn: CellEnergy | EnergyFn,
        *,
        pin_disp: npt.ArrayLike | None = None,
        pin_mask: npt.ArrayLike | None = None,
    ) -> None:
        self.model = Model(mesh, energy_fn)
        if pin_disp is not None:
            self.pin_disp = np.asarray(pin_disp, float)
        elif "pin_disp" in self.point_data:
            self.pin_disp = np.asarray(self.point_data["pin_disp"], float)
        else:
            self.pin_disp = np.zeros((self.n_points, 3), float)
        if pin_mask is not None:
            self.pin_mask = np.asarray(pin_mask, bool)
        elif "pin_mask" in self.point_data:
            self.pin_mask = np.asarray(self.point_data["pin_mask"], bool)
        else:
            self.pin_mask = np.zeros((self.n_points,), bool)

    def solve(self) -> scipy.optimize.OptimizeResult:
        x0: npt.NDArray[np.floating] = np.zeros((self.n_free, 3))
        try:
            res: scipy.optimize.OptimizeResult = scipy.optimize.minimize(
                self.fun,
                x0.flatten(),
                method="trust-constr",
                jac=self.jac,
                hess=self.hess,
                options={"disp": True},
            )
        except Exception as e:
            res = scipy.optimize.OptimizeResult()
            res["execution_time"] = np.nan
            res["message"] = str(e)
            res["success"] = False
            res["x"] = x0.flatten()
        return res

    def fun(self, x: npt.ArrayLike) -> jax.Array:
        disp: npt.NDArray[np.floating] = self.make_disp(x)
        energy: jax.Array = self.model.energy(disp)
        return energy

    def jac(self, x: npt.ArrayLike) -> jax.Array:
        disp: npt.NDArray[np.floating] = self.make_disp(x)
        jac: jax.Array = self.model.energy_jac(disp)
        return jac[self.free_mask].flatten()

    def hess(self, x: npt.ArrayLike) -> scipy.sparse.coo_matrix:
        disp: npt.NDArray[np.floating] = self.make_disp(x)
        hess: sparse.COO = self.model.energy_hess(disp)
        coord_mask: npt.NDArray[np.bool] = np.ones((3,), bool)
        hess = mkit.math.sparse.mask(
            hess, (self.free_mask, coord_mask, self.free_mask, coord_mask)
        )
        return hess.reshape((self.n_free * 3, self.n_free * 3)).to_scipy_sparse()

    def make_disp(self, x: npt.ArrayLike) -> npt.NDArray[np.floating]:
        disp: npt.NDArray[np.floating] = np.empty((self.n_points, 3))
        disp[self.free_mask] = np.reshape(x, (self.n_free, 3))
        disp[self.pin_mask] = self.pin_disp[self.pin_mask]
        return disp

    @property
    def free_mask(self) -> npt.NDArray[np.bool]:
        return ~self.pin_mask

    @property
    def n_free(self) -> int:
        return np.count_nonzero(self.free_mask)

    @property
    def n_points(self) -> int:
        return self.model.n_points

    @property
    def n_pin(self) -> int:
        return np.count_nonzero(self.pin_mask)

    @property
    def point_data(self) -> pv.DataSetAttributes:
        return self.model.point_data
