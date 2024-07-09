import jax
import jax.numpy as jnp
import jax.typing as jt


def calc_volume(points: jt.ArrayLike) -> jax.Array:
    points = jnp.asarray(points)  # (4, 3) float
    shape: jax.Array = jnp.vstack(
        [points[i] - points[3] for i in range(3)]
    )  # (3, 3) float
    volume: jax.Array = jnp.abs(jnp.linalg.det(shape)) / 6  # () float
    return volume


def calc_grad(points: jt.ArrayLike) -> jax.Array:
    points = jnp.asarray(points)  # (4, 3) float
    shape: jax.Array = jnp.vstack(
        [points[i] - points[3] for i in range(3)]
    )  # (3, 3) float
    shape_inv: jax.Array = jnp.linalg.inv(shape)  # (3, 3) float
    grad: jax.Array = jnp.hstack(
        [shape_inv, -shape_inv @ jnp.ones((3, 1))]
    )  # (3, 4) float
    return grad


def calc_deformation_gradient(
    position: jt.ArrayLike, position_rest: jt.ArrayLike
) -> jax.Array:
    position = jnp.asarray(position)  # (4, 3) float
    position_rest = jnp.asarray(position_rest)  # (4, 3) float
    grad: jax.Array = calc_grad(position_rest)  # (3, 4) float
    deformation_gradient: jax.Array = grad @ position  # (3, 3) float
    return deformation_gradient
