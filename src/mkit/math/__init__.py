from typing import Literal

import jax
import jax.numpy as jnp
import jax.typing as jxt


def frobenius_norm_square(_x: jxt.ArrayLike) -> jax.Array:
    x: jax.Array = jnp.asarray(_x)
    return jnp.sum(x**2)


F2 = frobenius_norm_square


def polar(
    a: jxt.ArrayLike,
    side: Literal["left", "right"] = "left",
    *args,
    method: Literal["svd", "qdwh"] = "svd",
    **kwargs,
) -> tuple[jax.Array, jax.Array]:
    """Computes the polar decomposition."""
    return jax.scipy.linalg.polar(
        jax.lax.stop_gradient(a), side, *args, method=method, **kwargs
    )
