import jax.numpy as jnp

import mkit.typing.jax as tj


def volume(points: tj.F43Like) -> tj.F:
    points: tj.F43 = jnp.asarray(points)
    shape: tj.F33 = jnp.vstack([points[i] - points[3] for i in range(3)])
    volume: tj.F = jnp.abs(jnp.linalg.det(shape)) / 6
    return volume
