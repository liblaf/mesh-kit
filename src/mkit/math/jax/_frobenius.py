import jax
import jax.numpy as jnp
import jax.typing as jxt

import mkit.typing.jax as jt


def frobenius_norm_square(x: jxt.ArrayLike) -> jt.D:
    x: jax.Array = jnp.asarray(x)
    return jnp.sum(x**2)


F2 = frobenius_norm_square
