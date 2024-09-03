import jax
import jax.numpy as jnp
import jax.typing as jxt


def invariants(_A: jxt.ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    A: jax.Array = jnp.asarray(_A)
    I1: jax.Array = jnp.trace(A)
    I2: jax.Array = 0.5 * (jnp.trace(A) ** 2 - jnp.trace(A @ A))
    I3: jax.Array = jnp.linalg.det(A)
    return I1, I2, I3
