from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import jax.typing as jxt


def gravity(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike] = {},
    cell_data: Mapping[str, jxt.ArrayLike] = {},
    field_data: Mapping[str, jxt.ArrayLike] = {},
) -> jax.Array:
    _: Any
    disp = jnp.asarray(disp)
    disp_center: jax.Array = jnp.mean(disp, axis=0)
    rho: jax.Array = jnp.asarray(cell_data["density"])
    g: jax.Array = jnp.asarray(cell_data["gravity"])
    W: jax.Array = -rho * jnp.dot(disp_center, g)
    return W
