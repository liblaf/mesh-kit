import jax
import jax.numpy as jnp
import jax.typing as jxt


def volume(_points: jxt.ArrayLike) -> jax.Array:
    points: jax.Array = jnp.asarray(_points)  # (4, 3)
    shape: jax.Array = jnp.vstack([points[i] - points[3] for i in range(3)])  # (3, 3)
    volume: jax.Array = jnp.abs(jnp.linalg.det(shape)) / 6  # ()
    return volume
