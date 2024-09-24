import jax
import jax.numpy as jnp
from jaxtyping import Float

import mkit.typing.jax as jt


def frobenius_norm_square(x: Float[jt.ArrayLike, "..."]) -> jt.F:
    x: jax.Array = jnp.asarray(x)
    return jnp.sum(x**2)
