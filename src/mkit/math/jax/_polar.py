from typing import Literal

import jax

import mkit.typing.jax as jt


def polar(
    a: jt.FMNLike,
    side: Literal["left", "right"] = "left",
    *args,
    method: Literal["svd", "qdwh"] = "svd",
    **kwargs,
) -> tuple[jt.FMN, jt.FNN]:
    """Computes the polar decomposition."""
    return jax.scipy.linalg.polar(
        jax.lax.stop_gradient(a), side, *args, method=method, **kwargs
    )
