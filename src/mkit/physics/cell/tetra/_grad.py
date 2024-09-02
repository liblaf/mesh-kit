import jax
import jax.numpy as jnp
import jax.typing as jxt


def grad_op(points: jxt.ArrayLike) -> jax.Array:
    X: jax.Array = jnp.asarray(points)  # (4, 3)
    shape: jax.Array = jnp.vstack([X[i] - X[3] for i in range(3)])  # (3, 3)
    shape_inv: jax.Array = jnp.linalg.pinv(shape)  # (3, 3)
    grad_op: jax.Array = jnp.hstack(
        [shape_inv, -jnp.sum(shape_inv, axis=1).reshape(3, 1)]
    )  # (3, 4)
    return grad_op
