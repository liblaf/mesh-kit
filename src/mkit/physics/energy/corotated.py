from typing import Any, no_type_check

import jax
import jax.numpy as jnp
import jax.typing as jt
import numpy as np
import numpy.typing as npt
import scipy
import scipy.sparse
import taichi as ti

import mkit.io
from mkit.physics.common import element
from mkit.physics.energy import base


def calc_corotated_element(
    position: jt.ArrayLike,
    position_rest: jt.ArrayLike,
    *,
    mu: float = 1,
    lambda_: float = 3,
) -> jax.Array:
    volume: jax.Array = element.calc_volume(position_rest)
    deformation_gradient: jax.Array = element.calc_deformation_gradient(
        position, position_rest
    )
    scaling: jax.Array
    rotation: jax.Array
    scaling, rotation = jax.scipy.linalg.polar(
        jax.lax.stop_gradient(deformation_gradient)
    )
    return volume * (
        mu * jnp.sum((deformation_gradient - rotation) ** 2)
        + lambda_ * (jnp.linalg.det(deformation_gradient) - 1) ** 2
    )


calc_corotated = jax.vmap(calc_corotated_element)
calc_corotated_element_jac = base.energy_element_jac(calc_corotated_element)
calc_corotated_jac = jax.vmap(calc_corotated_element_jac)
calc_corotated_element_hess = base.energy_element_hess(calc_corotated_element)
calc_corotated_hess = jax.vmap(calc_corotated_element_hess)


@ti.data_oriented
class Corotated(base.Energy):
    def __init__(self, mesh: Any) -> None:
        self.mesh = mkit.io.as_taichi(mesh, ["CV"])
        self.mesh.verts.place({"position": ti.math.vec3, "jac": ti.math.vec3})
        self.mesh.cells.place(
            {
                "jac": ti.types.matrix(4, 3, float),
                "hess": ti.types.matrix(12, 12, float),
            }
        )
        self.tetra = jnp.asarray(mkit.io.as_meshio(mesh).get_cells_type("tetra"))

    def calc_energy(self) -> jax.Array:
        return calc_corotated(
            self.position.to_numpy()[self.tetra], self.position_rest[self.tetra]
        ).sum()

    def calc_energy_jac(self) -> npt.NDArray[np.floating]:
        jac_cells: ti.MatrixField = self.mesh.cells.get_member_field("jac")
        jac_cells.from_numpy(
            np.asarray(
                calc_corotated_jac(
                    self.position.to_numpy()[self.tetra], self.position_rest[self.tetra]
                )
            )
        )
        jac: ti.MatrixField = self.mesh.verts.get_member_field("jac")
        jac.fill(0)
        self.assemble_jac()
        return jac.to_numpy()  # pyright: ignore [reportReturnType]

    def calc_energy_hess(self) -> scipy.sparse.coo_array:
        hess_cells: ti.MatrixField = self.mesh.cells.get_member_field("hess")
        hess_cells.from_numpy(
            np.asarray(
                calc_corotated_hess(
                    self.position.to_numpy()[self.tetra], self.position_rest[self.tetra]
                ).reshape(self.n_cells, 12, 12)
            )
        )
        coords: ti.MatrixField = ti.field(int, (2, 144 * self.n_cells))
        self.assemble_hess(coords)
        hess = scipy.sparse.coo_array(
            (hess_cells.to_numpy().flatten(), coords.to_numpy()),
            shape=(self.n_verts * 3, self.n_verts * 3),
        )
        hess.sum_duplicates()
        hess.eliminate_zeros()
        return hess

    @no_type_check
    @ti.kernel
    def assemble_jac(self):
        for c in self.mesh.cells:
            for i in ti.static(range(4)):
                c.verts[i].jac += c.jac[i, :]

    @no_type_check
    @ti.kernel
    def assemble_hess(self, coords: ti.template()):
        for c in self.mesh.cells:
            for i, j in ti.ndrange(12, 12):
                idx = 144 * c.id + 12 * i + j
                coords[0, idx] = 3 * c.verts[i // 3].id + i % 3
                coords[1, idx] = 3 * c.verts[j // 3].id + j % 3
