import jax
import jax.numpy as jnp
import jax.typing as jxt


def volume(points: jxt.ArrayLike) -> jax.Array:
    points = jnp.asarray(points)  # (4, 3)
    shape: jax.Array = jnp.vstack([points[i] - points[3] for i in range(3)])  # (3, 3)
    volume: jax.Array = jnp.abs(jnp.linalg.det(shape)) / 6  # ()
    return volume


def grad_op(points: jxt.ArrayLike) -> jax.Array:
    points = jnp.asarray(points)  # (4, 3)
    shape: jax.Array = jnp.vstack([points[i] - points[3] for i in range(3)])  # (3, 3)
    shape_inv: jax.Array = jnp.linalg.pinv(shape)  # (3, 3)
    grad_op: jax.Array = jnp.hstack(
        [shape_inv, -jnp.sum(shape_inv, axis=1).reshape(3, 1)]
    )  # (3, 4)
    return grad_op


def deformation_gradient(disp: jxt.ArrayLike, points: jxt.ArrayLike) -> jax.Array:
    disp = jnp.asarray(disp)  # (4, 3)
    points = jnp.asarray(points)  # (4, 3)
    grad_op_: jax.Array = grad_op(points)  # (3, 4)
    grad_u: jax.Array = grad_op_ @ disp  # (3, 3)
    F: jax.Array = grad_u + jnp.eye(3)  # (3, 3)
    return F
