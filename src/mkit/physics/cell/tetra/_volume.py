import jax.numpy as jnp

import mkit.typing.jax as j


def volume(_points: j.D43Like) -> j.D:
    points: j.D43 = jnp.asarray(_points)
    shape: j.D33 = jnp.vstack([points[i] - points[3] for i in range(3)])
    volume: j.D = jnp.abs(jnp.linalg.det(shape)) / 6
    return volume
