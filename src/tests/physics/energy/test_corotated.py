import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.typing as jt
import mkit.ext
import numpy as np
import taichi as ti
import trimesh
from mkit.physics.energy import corotated

if TYPE_CHECKING:
    import meshio
    import scipy.sparse


@functools.cache
def create_tetmesh() -> corotated.Corotated:
    ti.init()
    surface: trimesh.Trimesh = trimesh.creation.box()
    tetmesh: meshio.Mesh = mkit.ext.tetgen(surface)
    model = corotated.Corotated(tetmesh)
    model.position.from_numpy(tetmesh.points)
    return model


def calc_energy_naive(
    position: jt.ArrayLike, position_rest: jt.ArrayLike, tetra: jt.ArrayLike
) -> jax.Array:
    position = jnp.array(position)
    position_rest = jnp.array(position_rest)
    energy: jax.Array = corotated.calc_corotated(
        position[tetra], position_rest[tetra]
    ).sum()
    return energy


def test_energy_element() -> None:
    key: jax.Array = jax.random.key(0)
    points: jax.Array = jax.random.uniform(key, (4, 3))
    energy: jax.Array = corotated.calc_corotated_element(points, points)
    np.testing.assert_allclose(energy, jnp.zeros(()), rtol=0, atol=1e-32)


def test_energy_element_jac() -> None:
    key: jax.Array = jax.random.key(0)
    points: jax.Array = jax.random.uniform(key, (4, 3))
    energy_jac: jax.Array = corotated.calc_corotated_element_jac(points, points)
    np.testing.assert_allclose(energy_jac, jnp.zeros((4, 3)), rtol=0, atol=1e-16)


def test_energy_random() -> None:
    key: jax.Array = jax.random.key(0)
    model: corotated.Corotated = create_tetmesh()
    displacement: jax.Array = jax.random.uniform(key, (model.n_verts, 3))
    model.position.from_numpy(np.asarray(model.position_rest + displacement))
    energy_actual: jax.Array = model.calc_energy()
    energy_desired: jax.Array = calc_energy_naive(
        model.position.to_numpy(), model.position_rest, model.tetra
    )
    np.testing.assert_allclose(energy_actual, energy_desired, rtol=0, atol=0)


def test_energy_zero() -> None:
    model: corotated.Corotated = create_tetmesh()
    model.position.from_numpy(np.asarray(model.position_rest))
    energy_actual: jax.Array = model.calc_energy()
    np.testing.assert_allclose(energy_actual, 0, rtol=0, atol=0)


def test_energy_jac_random() -> None:
    key: jax.Array = jax.random.key(0)
    model: corotated.Corotated = create_tetmesh()
    displacement: jax.Array = jax.random.uniform(key, (model.n_verts, 3))
    model.position.from_numpy(np.asarray(model.position_rest + displacement))
    energy_jac_actual: jax.Array = model.calc_energy_jac()
    energy_jac_desired: jax.Array = jax.jacobian(calc_energy_naive)(
        model.position.to_numpy(), model.position_rest, model.tetra
    )
    np.testing.assert_allclose(
        energy_jac_actual, energy_jac_desired, rtol=1e-15, atol=1e-15
    )


def test_energy_jac_zero() -> None:
    model: corotated.Corotated = create_tetmesh()
    model.position.from_numpy(model.position_rest)
    np.testing.assert_allclose(
        model.calc_energy_jac(), jnp.zeros((model.n_verts, 3)), rtol=0, atol=0
    )


def test_energy_hess_random() -> None:
    key: jax.Array = jax.random.key(0)
    model: corotated.Corotated = create_tetmesh()
    displacement: jax.Array = jax.random.uniform(key, (model.n_verts, 3))
    model.position.from_numpy(np.asarray(model.position_rest + displacement))
    energy_hess_actual: scipy.sparse.coo_array = model.calc_energy_hess()
    energy_hess_desired: jax.Array = jax.hessian(calc_energy_naive)(
        model.position.to_numpy(), model.position_rest, model.tetra
    )
    np.testing.assert_allclose(
        energy_hess_actual.todense().reshape((model.n_verts, 3, model.n_verts, 3)),
        energy_hess_desired,
        rtol=1e-15,
        atol=1e-14,
    )


def test_energy_hess_zero() -> None:
    model: corotated.Corotated = create_tetmesh()
    model.position.from_numpy(np.asarray(model.position_rest))
    energy_hess_actual: scipy.sparse.coo_array = model.calc_energy_hess()
    energy_hess_desired: jax.Array = jax.hessian(calc_energy_naive)(
        model.position.to_numpy(), model.position_rest, model.tetra
    )
    np.testing.assert_allclose(
        energy_hess_actual.todense().reshape((model.n_verts, 3, model.n_verts, 3)),
        energy_hess_desired,
        rtol=0,
        atol=0,
    )
