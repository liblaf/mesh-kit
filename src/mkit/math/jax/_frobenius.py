import jax
import jax.numpy as jnp
import jax.typing as jxt


def frobenius_norm_square(_x: jxt.ArrayLike) -> jax.Array:
    x: jax.Array = jnp.asarray(_x)
    return jnp.sum(x**2)


F2 = frobenius_norm_square
