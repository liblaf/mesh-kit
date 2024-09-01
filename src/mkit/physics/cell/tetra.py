import jax
import jax.numpy as jnp
import jax.typing as jxt


def volume(_points: jxt.ArrayLike) -> jax.Array:
    points: jax.Array = jnp.asarray(_points)  # (4, 3)
    shape: jax.Array = jnp.vstack([points[i] - points[3] for i in range(3)])  # (3, 3)
    volume: jax.Array = jnp.abs(jnp.linalg.det(shape)) / 6  # ()
    return volume


def grad_op(_points: jxt.ArrayLike) -> jax.Array:
    points: jax.Array = jnp.asarray(_points)  # (4, 3)
    shape: jax.Array = jnp.vstack([points[i] - points[3] for i in range(3)])  # (3, 3)
    shape_inv: jax.Array = jnp.linalg.pinv(shape)  # (3, 3)
    grad_op: jax.Array = jnp.hstack(
        [shape_inv, -jnp.sum(shape_inv, axis=1).reshape(3, 1)]
    )  # (3, 4)
    return grad_op


def deformation_gradient(_disp: jxt.ArrayLike, _points: jxt.ArrayLike) -> jax.Array:
    disp: jax.Array = jnp.asarray(_disp)  # (4, 3)
    points: jax.Array = jnp.asarray(_points)  # (4, 3)
    _grad_op: jax.Array = grad_op(points)  # (3, 4)
    grad_u: jax.Array = _grad_op @ disp  # (3, 3)
    F: jax.Array = grad_u + jnp.eye(3)  # (3, 3)
    return F
