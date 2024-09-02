from collections.abc import Mapping

import jax
import jax.numpy as jnp
import jax.typing as jxt

from mkit.physics.cell import tetra
from mkit.physics.energy import cell_energy


@cell_energy
def linear(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    u: jax.Array = jnp.asarray(disp)  # (4, 3)
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])
    mu: jax.Array = jnp.asarray(cell_data["mu"])
    grad_op: jax.Array = tetra.grad_op(points)  # (3, 4)
    grad_u: jax.Array = grad_op @ u  # (3, 3)
    E: jax.Array = 0.5 * (grad_u.T + grad_u)  # (3, 3)
    W: jax.Array = 0.5 * lambda_ * jnp.trace(E) ** 2 + mu * jnp.trace(E @ E)
    return W
