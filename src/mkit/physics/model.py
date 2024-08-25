import functools
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import jax.typing as jxt
import numpy as np
import numpy.typing as npt
import pyvista as pv
import sparse

import mkit.io
from mkit.logging import log_time
from mkit.physics import utils
from mkit.physics.cell import tetra
from mkit.physics.energy.abc import CellEnergy


class Model:
    mesh: pv.UnstructuredGrid

    def __init__(self, mesh: Any) -> None:
        self.mesh = mkit.io.as_unstructured_grid(mesh)

    @log_time
    def energy(self, energy_fn: CellEnergy, disp: jxt.ArrayLike) -> jax.Array:
        W: jax.Array = self.energy_density(energy_fn, disp)  # (C,)
        return jnp.sum(self.cell_volume * W)

    @log_time
    def energy_density(self, energy_fn: CellEnergy, disp: jxt.ArrayLike) -> jax.Array:
        disp = jnp.asarray(disp)  # (V, 3)
        disp_mapped: jax.Array = disp[self.tetra]  # (C, 4, 3)
        return energy_fn.vmap(
            disp_mapped, self.points_mapped, self.point_data_mapped, self.cell_data
        )

    @log_time
    def energy_jac(self, energy_fn: CellEnergy, disp: jxt.ArrayLike) -> jax.Array:
        disp = jnp.asarray(disp)  # (V, 3)
        disp_mapped: jax.Array = disp[self.tetra]  # (C, 4, 3)
        jac_mapped: jax.Array = energy_fn.vmap_jac(
            disp_mapped, self.points_mapped, self.point_data_mapped, self.cell_data
        )  # (C, 4, 3)
        jac_mapped = self.cell_volume[:, None, None] * jac_mapped
        jac: jax.Array = jax.ops.segment_sum(
            jac_mapped.reshape((self.n_cells * 4, 3)),
            self.tetra.flatten(),
            num_segments=self.n_points,
        )  # (V, 3)
        return jac

    @log_time
    def energy_hess(self, energy_fn: CellEnergy, disp: jxt.ArrayLike) -> sparse.COO:
        disp = jnp.asarray(disp)  # (V, 3)
        disp_mapped: jax.Array = disp[self.tetra]  # (C, 4, 3)
        hess_mapped: jax.Array = energy_fn.vmap_hess(
            disp_mapped, self.points_mapped, self.point_data_mapped, self.cell_data
        )  # (C, 4, 3, 4, 3)
        hess_mapped = self.cell_volume[:, None, None, None, None] * hess_mapped
        return sparse.COO(
            self.energy_hess_coords,
            hess_mapped.flatten(),
            shape=(self.n_points, 3, self.n_points, 3),
            prune=True,
        )

    @functools.cached_property
    def energy_hess_coords(self) -> npt.NDArray[np.integer]:
        return utils.hess_coords(self.tetra)

    @property
    def cell_data(self) -> Mapping[str, jxt.ArrayLike]:
        return dict(self.mesh.cell_data)  # pyright: ignore [reportReturnType]

    @functools.cached_property
    def cell_volume(self) -> jax.Array:
        """(C,)."""
        return jax.jit(jax.vmap(tetra.volume))(self.points_mapped)

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    @property
    def point_data(self) -> Mapping[str, jxt.ArrayLike]:
        return self.mesh.point_data  # pyright: ignore [reportReturnType]

    @functools.cached_property
    def point_data_mapped(self) -> dict[str, jax.Array]:
        return {k: jnp.asarray(v)[self.tetra] for k, v in self.point_data.items()}

    @functools.cached_property
    def points(self) -> jax.Array:
        return jnp.asarray(self.mesh.points)

    @functools.cached_property
    def points_mapped(self) -> jax.Array:
        return self.points[self.tetra]

    @functools.cached_property
    def tetra(self) -> npt.NDArray[np.integer]:
        return self.mesh.cells_dict[pv.CellType.TETRA]
