import jax.numpy as jnp

import mkit.typing.jax as jt


def volume(points: jt.F43Like) -> jt.F:
    points: jt.F43 = jnp.asarray(points)
    shape: jt.F33 = jnp.vstack([points[i] - points[3] for i in range(3)])
    volume: jt.F = jnp.abs(jnp.linalg.det(shape)) / 6
    return volume
