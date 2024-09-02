from typing import Literal

import jax
import jax.typing as jxt


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
