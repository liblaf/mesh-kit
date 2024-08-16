from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.typing as jxt
import mkit.ext
import numpy as np
import pytest
import pyvista as pv
import taichi as ti
from mkit.physics.energy.abc import CellEnergyFn
from mkit.physics.energy.elastic import corotated
from mkit.physics.model import Model

if TYPE_CHECKING:
    import meshio
    import scipy.sparse
    import sparse


def energy_naive(disp: jxt.ArrayLike, model: Model) -> jax.Array:
    disp = jnp.array(disp)
    W: jax.Array = jax.vmap(corotated)(
        disp[model.tetra], model.points_mapped, cell_data=model.cell_data
    )
    return jnp.sum(model.cell_volume * W)


@pytest.fixture()
def disp_zero(model: Model) -> jax.Array:
    return jnp.zeros((model.n_points, 3))


@pytest.fixture()
def disp_random(model: Model) -> jax.Array:
    key: jax.Array = jax.random.key(0)
    return jax.random.uniform(key, (model.n_points, 3))


@pytest.fixture()
def energy_fn() -> CellEnergyFn:
    return CellEnergyFn(corotated)


@pytest.fixture()
def model() -> Model:
    ti.init()
    surface: pv.PolyData = pv.Box()
    mesh: meshio.Mesh = mkit.ext.tetgen(surface)
    n_cells: int = len(mesh.get_cells_type("tetra"))
    mesh.cell_data = {  # pyright: ignore [reportAttributeAccessIssue]
        "mu": [jnp.full((n_cells,), 1.0)],
        "lambda": [jnp.full((n_cells,), 3.0)],
    }
    model = Model(mesh)
    return model


def test_cell_energy(energy_fn: CellEnergyFn) -> None:
    key: jax.Array = jax.random.key(0)
    disp: jax.Array = jnp.zeros((4, 3))
    points: jax.Array = jax.random.uniform(key, (4, 3))
    energy: jax.Array = energy_fn(disp, points, cell_data={"mu": 1.0, "lambda": 3.0})
    np.testing.assert_allclose(energy, 0, rtol=0, atol=0)


def test_cell_energy_jac(energy_fn: CellEnergyFn) -> None:
    key: jax.Array = jax.random.key(0)
    disp: jax.Array = jnp.zeros((4, 3))
    points: jax.Array = jax.random.uniform(key, (4, 3))
    energy_jac: jax.Array = energy_fn.jac(
        disp, points, cell_data={"mu": 1.0, "lambda": 3.0}
    )
    np.testing.assert_allclose(energy_jac, jnp.zeros((4, 3)), rtol=0, atol=0)


def test_energy_random(
    model: Model,
    energy_fn: CellEnergyFn,
    disp_random: jax.Array,
) -> None:
    energy_actual: jax.Array = model.energy(energy_fn, disp_random)
    energy_desired: jax.Array = energy_naive(disp_random, model)
    np.testing.assert_allclose(energy_actual, energy_desired, rtol=0, atol=0)


def test_energy_zero(
    model: Model,
    energy_fn: CellEnergyFn,
    disp_zero: jax.Array,
) -> None:
    energy_actual: jax.Array = model.energy(energy_fn, disp_zero)
    np.testing.assert_allclose(energy_actual, 0, rtol=0, atol=0)


def test_energy_jac_random(
    model: Model,
    energy_fn: CellEnergyFn,
    disp_random: jax.Array,
) -> None:
    jac_actual: jax.Array = model.energy_jac(energy_fn, disp_random)
    jac_desired: jax.Array = jax.jacobian(energy_naive)(disp_random, model)
    np.testing.assert_allclose(jac_actual, jac_desired, rtol=1e-14, atol=0)


def test_energy_jac_zero(
    model: Model,
    energy_fn: CellEnergyFn,
    disp_zero: jax.Array,
) -> None:
    jac_actual: jax.Array = model.energy_jac(energy_fn, disp_zero)
    jac_desired: jax.Array = jnp.zeros((model.n_points, 3))
    np.testing.assert_allclose(jac_actual, jac_desired, rtol=0, atol=0)


def test_energy_hess_random(
    model: Model,
    energy_fn: CellEnergyFn,
    disp_random: jax.Array,
) -> None:
    hess_actual: sparse.COO = model.energy_hess(energy_fn, disp_random)
    hess_desired: jax.Array = jax.hessian(energy_naive)(disp_random, model)
    np.testing.assert_allclose(hess_actual.todense(), hess_desired, rtol=1e-13, atol=0)


def test_energy_hess_zero(
    model: Model,
    energy_fn: CellEnergyFn,
    disp_zero: jax.Array,
) -> None:
    energy_hess_actual: scipy.sparse.coo_array = model.energy_hess(energy_fn, disp_zero)
    energy_hess_desired: jax.Array = jax.hessian(energy_naive)(disp_zero, model)
    np.testing.assert_allclose(
        energy_hess_actual.todense(), energy_hess_desired, rtol=0, atol=1e-14
    )
