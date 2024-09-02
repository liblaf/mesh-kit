import functools
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import jax.typing as jxt
import numpy as np
import numpy.typing as npt
import pyvista as pv
import sparse
from jax.numpy import newaxis

import mkit
from mkit.io import AnyTetMesh
from mkit.physics.energy import CellEnergy, EnergyFn
from mkit.physics.model._hess import hess_coords


class Model:
    mesh: pv.UnstructuredGrid
    energy_fn: CellEnergy

    def __init__(self, mesh: AnyTetMesh, energy_fn: CellEnergy | EnergyFn) -> None:
        self.mesh = mkit.io.as_unstructured_grid(mesh)
        self.energy_fn = CellEnergy(energy_fn)

    def energy(self, disp: jxt.ArrayLike) -> jax.Array:
        W: jax.Array = self.energy_density(disp)
        return jnp.dot(W, self.cell_volume)

    def energy_density(self, disp: jxt.ArrayLike) -> jax.Array:
        W: jax.Array = self.energy_fn.vmap(self.make_disp_vmap(disp), *self._args)
        return W

    def energy_jac(self, disp: jxt.ArrayLike) -> jax.Array:
        jac_vmap: jax.Array = self.energy_fn.jac_vmap(
            self.make_disp_vmap(disp), *self._args
        )  # (C, 4, 3)
        jac_vmap *= self.cell_volume[:, newaxis, newaxis]
        jac: jax.Array = jax.ops.segment_sum(
            jac_vmap.reshape((self.n_cells * 4, 3)),
            self.tetra.flatten(),
            num_segments=self.n_points,
        )  # (V, 3)
        return jac

    def energy_hess(self, disp: jxt.ArrayLike) -> sparse.COO:
        hess_vmap: jax.Array = self.energy_fn.hess_vmap(
            self.make_disp_vmap(disp), *self._args
        )  # (C, 4, 3, 4, 3)
        hess_vmap *= self.cell_volume[:, newaxis, newaxis, newaxis, newaxis]
        return sparse.COO(
            self.energy_hess_coords,
            hess_vmap.flatten(),
            shape=(self.n_points, 3, self.n_points, 3),
            prune=True,
        )

    @functools.cached_property
    def energy_hess_coords(self) -> npt.NDArray[np.integer]:
        return hess_coords(self.tetra)

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    @functools.cached_property
    def _args(
        self,
    ) -> tuple[
        jxt.ArrayLike,
        Mapping[str, jxt.ArrayLike],
        Mapping[str, jxt.ArrayLike],
        Mapping[str, jxt.ArrayLike],
    ]:
        return (
            self.points_vmap,
            self.point_data_vmap,
            dict(self.cell_data),
            dict(self.field_data),
        )

    @property
    def points_vmap(self) -> npt.NDArray[np.floating]:
        return self.points[self.tetra]

    def make_disp_vmap(self, disp: jxt.ArrayLike) -> jax.Array:
        u: jax.Array = jnp.asarray(disp)  # (V, 3)
        u_vmap: jax.Array = u[self.tetra]  # (C, 4, 3)
        return u_vmap

    @property
    def cell_volume(self) -> npt.NDArray[np.floating]:
        if "Volume" not in self.cell_data:
            self.mesh = self.mesh.compute_cell_sizes(
                length=False,
                area=False,
                volume=True,
                progress_bar=True,
                vertex_count=False,
            )  # pyright: ignore [reportAttributeAccessIssue]
        volume: npt.NDArray[np.floating] = np.abs(self.cell_data["Volume"])
        return volume

    @property
    def points(self) -> npt.NDArray[np.floating]:
        return self.mesh.points

    @property
    def tetra(self) -> npt.NDArray[np.integer]:
        return self.mesh.cells_dict[pv.CellType.TETRA]

    @property
    def point_data(self) -> pv.DataSetAttributes:
        return self.mesh.point_data

    @property
    def cell_data(self) -> pv.DataSetAttributes:
        return self.mesh.cell_data

    @property
    def field_data(self) -> pv.DataSetAttributes:
        return self.mesh.field_data

    @property
    def point_data_vmap(self) -> dict[str, jax.Array]:
        return {k: jnp.asarray(v)[self.tetra] for k, v in self.point_data.items()}
