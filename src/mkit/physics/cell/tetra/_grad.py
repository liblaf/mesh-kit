import jax.numpy as jnp

import mkit.typing.jax as jt


def grad_op(points: jt.F43Like) -> jt.F34:
    X: jt.F43 = jnp.asarray(points)
    shape: jt.F33 = jnp.vstack([X[i] - X[3] for i in range(3)])
    shape_inv: jt.F33 = jnp.linalg.pinv(shape)
    grad_op: jt.F34 = jnp.hstack([shape_inv, -jnp.sum(shape_inv, axis=1).reshape(3, 1)])
    return grad_op
