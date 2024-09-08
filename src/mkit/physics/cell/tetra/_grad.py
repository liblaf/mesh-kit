import jax.numpy as jnp

import mkit.typing.jax as j


def grad_op(points: j.D43Like) -> j.D34:
    X: j.D43 = jnp.asarray(points)
    shape: j.D33 = jnp.vstack([X[i] - X[3] for i in range(3)])
    shape_inv: j.D33 = jnp.linalg.pinv(shape)
    grad_op: j.D34 = jnp.hstack([shape_inv, -jnp.sum(shape_inv, axis=1).reshape(3, 1)])
    return grad_op
