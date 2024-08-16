import functools
from typing import Any, no_type_check

import jax
import jax.numpy as jnp
import jax.typing as jxt
import meshio
import numpy as np
import numpy.typing as npt
import sparse
import taichi as ti

import mkit.io
import mkit.logging
from mkit.physics.cell import tetra
from mkit.physics.energy.abc import CellEnergyFn


class Model:
    mesh: meshio.Mesh

    def __init__(self, mesh: Any) -> None:
        self.mesh = mkit.io.as_meshio(mesh)

    @mkit.logging.log_time
    def energy(self, energy_fn: CellEnergyFn, disp: jxt.ArrayLike) -> jax.Array:
        W: jax.Array = self.energy_density(energy_fn, disp)  # (C,)
        return jnp.sum(self.cell_volume * W)

    @mkit.logging.log_time
    def energy_density(self, energy_fn: CellEnergyFn, disp: jxt.ArrayLike) -> jax.Array:
        disp = jnp.asarray(disp)  # (V, 3)
        disp_mapped: jax.Array = disp[self.tetra]  # (C, 4, 3)
        return energy_fn.vmap(
            disp_mapped, self.points_mapped, self.point_data_mapped, self.cell_data
        )

    @mkit.logging.log_time
    def energy_jac(self, energy_fn: CellEnergyFn, disp: jxt.ArrayLike) -> jax.Array:
        disp = jnp.asarray(disp)  # (V, 3)
        disp_mapped: jax.Array = disp[self.tetra]  # (C, 4, 3)
        jac_mapped: jax.Array = energy_fn.vmap_jac(
            disp_mapped, self.points_mapped, self.point_data_mapped, self.cell_data
        )  # (C, 4, 3)
        jac_mapped = self.cell_volume[:, None, None] * jac_mapped
        jac: jax.Array = jax.ops.segment_sum(
            jac_mapped.reshape((-1, 3)),
            self.tetra.flatten(),
            num_segments=self.n_points,
        )
        return jac

    @functools.cached_property
    def energy_jac_coords(self) -> npt.NDArray[np.integer]:
        mesh: ti.MeshInstance = mkit.io.as_taichi(self.mesh, ["CV"])
        coords: ti.ScalarField = ti.field(int, (2, 12 * len(self.tetra)))

        @no_type_check
        @ti.kernel
        def init_coords(mesh: ti.template(), coords: ti.template()):
            for c in mesh.cells:
                for u, i in ti.ndrange(4, 3):
                    idx = 12 * c.id + 3 * u + i
                    coords[0, idx] = c.verts[u].id
                    coords[1, idx] = i

        init_coords(mesh, coords)
        return coords.to_numpy()  # pyright: ignore [reportReturnType]

    @mkit.logging.log_time
    def energy_hess(self, energy_fn: CellEnergyFn, disp: jxt.ArrayLike) -> sparse.COO:
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
        mesh: ti.MeshInstance = mkit.io.as_taichi(self.mesh, ["CV"])
        coords: ti.ScalarField = ti.field(int, (4, 144 * len(self.tetra)))

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

    @functools.cached_property
    def cell_data(self) -> dict[str, jxt.ArrayLike]:
        return {k: v[0] for k, v in self.mesh.cell_data.items()}  # pyright: ignore [reportReturnType]

    @functools.cached_property
    def cell_volume(self) -> jax.Array:
        """(C,)."""
        return jax.vmap(tetra.volume)(self.points_mapped)

    @functools.cached_property
    def n_cells(self) -> int:
        return len(self.tetra)

    @functools.cached_property
    def n_points(self) -> int:
        return len(self.points)

    @functools.cached_property
    def point_data(self) -> dict[str, jxt.ArrayLike]:
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
        return self.mesh.get_cells_type("tetra")
