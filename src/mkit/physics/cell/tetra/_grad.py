import jax.numpy as jnp

import mkit.typing.jax as tj


def grad_op(points: tj.F43Like) -> tj.F34:
    X: tj.F43 = jnp.asarray(points)
    shape: tj.F33 = jnp.vstack([X[i] - X[3] for i in range(3)])
    shape_inv: tj.F33 = jnp.linalg.pinv(shape)
    grad_op: tj.F34 = jnp.hstack([shape_inv, -jnp.sum(shape_inv, axis=1).reshape(3, 1)])
    return grad_op
