from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.typing as jxt
import mkit.ext
import numpy as np
import pytest
from mkit.physics.energy import elastic
from mkit.physics.model import Model

if TYPE_CHECKING:
    import pyvista as pv
    import scipy.sparse
    import sparse


def energy_naive(disp: jxt.ArrayLike, model: Model) -> jax.Array:
    W: jax.Array = jax.jit(jax.vmap(elastic.linear.fn))(
        model.make_disp_vmap(disp), model.points_vmap, {}, dict(model.cell_data), {}
    )
    return jnp.dot(model.cell_volume, W)


@pytest.fixture()
def disp_zero(model: Model) -> jax.Array:
    return jnp.zeros((model.n_points, 3))


@pytest.fixture()
def disp_random(model: Model) -> jax.Array:
    key: jax.Array = jax.random.key(0)
    return jax.random.uniform(key, (model.n_points, 3))


@pytest.fixture()
def model() -> Model:
    mesh: pv.UnstructuredGrid = mkit.creation.tet.box()
    mesh.cell_data.update({"mu": 1.0, "lambda": 3.0})  # pyright: ignore [reportArgumentType]
    model = Model(mesh, elastic.linear)
    return model


def test_energy_random(model: Model, disp_random: jax.Array) -> None:
    energy_actual: jax.Array = model.energy(disp_random)
    energy_desired: jax.Array = energy_naive(disp_random, model)
    np.testing.assert_allclose(energy_actual, energy_desired, rtol=0, atol=0)


def test_energy_zero(model: Model, disp_zero: jax.Array) -> None:
    energy_actual: jax.Array = model.energy(disp_zero)
    np.testing.assert_allclose(energy_actual, 0, rtol=0, atol=0)


def test_energy_jac_random(model: Model, disp_random: jax.Array) -> None:
    jac_actual: jax.Array = model.energy_jac(disp_random)
    jac_desired: jax.Array = jax.jacobian(energy_naive)(disp_random, model)
    np.testing.assert_allclose(jac_actual, jac_desired, rtol=1e-15, atol=0)


def test_energy_jac_zero(model: Model, disp_zero: jax.Array) -> None:
    jac_actual: jax.Array = model.energy_jac(disp_zero)
    jac_desired: jax.Array = jnp.zeros((model.n_points, 3))
    np.testing.assert_allclose(jac_actual, jac_desired, rtol=0, atol=0)


def test_energy_hess_random(model: Model, disp_random: jax.Array) -> None:
    hess_actual: sparse.COO = model.energy_hess(disp_random)
    hess_desired: jax.Array = jax.hessian(energy_naive)(disp_random, model)
    np.testing.assert_allclose(hess_actual.todense(), hess_desired, rtol=1e-14, atol=0)


def test_energy_hess_zero(model: Model, disp_zero: jax.Array) -> None:
    energy_hess_actual: scipy.sparse.coo_array = model.energy_hess(disp_zero)
    energy_hess_desired: jax.Array = jax.hessian(energy_naive)(disp_zero, model)
    np.testing.assert_allclose(
        energy_hess_actual.todense(), energy_hess_desired, rtol=1e-14, atol=0
    )
