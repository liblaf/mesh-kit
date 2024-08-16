from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import jax.typing as jxt

from mkit.physics.cell import tetra


def linear(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike] = {},
    cell_data: Mapping[str, jxt.ArrayLike] = {},
    field_data: Mapping[str, jxt.ArrayLike] = {},
) -> jax.Array:
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])
    mu: jax.Array = jnp.asarray(cell_data["mu"])
    grad_op: jax.Array = tetra.grad_op(points)  # (3, 4)
    grad_u: jax.Array = grad_op @ disp  # (3, 3)
    E: jax.Array = 0.5 * (grad_u.T + grad_u)  # (3, 3)
    W: jax.Array = 0.5 * lambda_ * jnp.trace(E) ** 2 + mu * jnp.trace(E @ E)
    return W


def corotated(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike] = {},
    cell_data: Mapping[str, jxt.ArrayLike] = {},
    field_data: Mapping[str, jxt.ArrayLike] = {},
) -> jax.Array:
    _: Any
    F: jax.Array = tetra.deformation_gradient(disp, points)
    R: jax.Array
    _, R = jax.scipy.linalg.polar(jax.lax.stop_gradient(F))
    mu: jax.Array = jnp.asarray(cell_data["mu"])
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])
    W: jax.Array = mu * jnp.sum((F - R) ** 2) + lambda_ * (jnp.linalg.det(F) - 1) ** 2
    return W
