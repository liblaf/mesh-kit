import jax
import jax.numpy as jnp
from jaxtyping import Float

import mkit.typing.jax as tj


def frobenius_norm_square(x: Float[tj.ArrayLike, "..."]) -> tj.F:
    x: jax.Array = jnp.asarray(x)
    return jnp.sum(x**2)
