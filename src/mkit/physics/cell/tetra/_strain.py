import jax
import jax.numpy as jnp
import jax.typing as jxt

from mkit.physics.cell import tetra


def deformation_gradient(disp: jxt.ArrayLike, points: jxt.ArrayLike) -> jax.Array:
    u: jax.Array = jnp.asarray(disp)  # (4, 3)
    grad_op: jax.Array = tetra.grad_op(points)  # (3, 4)
    grad_u: jax.Array = grad_op @ u  # (3, 3)
    F: jax.Array = grad_u + jnp.eye(3)  # (3, 3)
    return F


def lagrangian_strain(disp: jxt.ArrayLike, points: jxt.ArrayLike) -> jax.Array:
    """Lagrangian finite strain tensor.

    Reference:
        1. <https://en.wikipedia.org/wiki/Finite_strain_theory#Finite_strain_tensors>
    """
    u: jax.Array = jnp.asarray(disp)
    grad_op: jax.Array = tetra.grad_op(points)  # (3, 4)
    grad_u: jax.Array = grad_op @ u  # (3, 3)
    E: jax.Array = 0.5 * (grad_u.T + grad_u + grad_u.T @ grad_u)
    return E
