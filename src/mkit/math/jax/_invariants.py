import jax
import jax.numpy as jnp

import mkit.typing.jax as jt


def invariants(A: jt.FNNLike) -> tuple[jt.F, jt.F, jt.F]:
    A: jax.Array = jnp.asarray(A)
    I1: jax.Array = jnp.trace(A)
    I2: jax.Array = 0.5 * (jnp.trace(A) ** 2 - jnp.trace(A @ A))
    I3: jax.Array = jnp.linalg.det(A)
    return I1, I2, I3
