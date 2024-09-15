from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import jax.typing as jxt
import matplotlib.pyplot as plt
import numpy as np

from mkit.physics.cell import tetra

E: float = 3e3
nu: float = 0.46
mu: float = E / (2 * (1 + nu))
lambda_: float = E * nu / ((1 + nu) * (1 - 2 * nu))


def linear(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike] = {},
    cell_data: Mapping[str, jxt.ArrayLike] = {},
    field_data: Mapping[str, jxt.ArrayLike] = {},
) -> jax.Array:
    grad_op: jax.Array = tetra.grad_op(points)
    grad_u: jax.Array = grad_op @ disp
    E: jax.Array = 0.5 * (grad_u.T + grad_u)
    W: jax.Array = 0.5 * lambda_ * jnp.trace(E) ** 2 + mu * jnp.trace(E @ E)
    return W


def neo_hookean(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike] = {},
    cell_data: Mapping[str, jxt.ArrayLike] = {},
    field_data: Mapping[str, jxt.ArrayLike] = {},
) -> jax.Array:
    F: jax.Array = tetra.deformation_gradient(disp, points)
    lambda_hat: jax.Array = mu + lambda_
    W: jax.Array = (
        0.5 * mu * jnp.sum(F**2)
        + 0.5 * lambda_hat * (jnp.linalg.det(F) - 1 - mu / lambda_hat) ** 2
    )
    return W


def nh(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike] = {},
    cell_data: Mapping[str, jxt.ArrayLike] = {},
    field_data: Mapping[str, jxt.ArrayLike] = {},
) -> jax.Array:
    F = tetra.deformation_gradient(disp, points)
    J = jnp.linalg.det(F)
    C = F.T @ F
    I_C = jnp.trace(C)
    W = 0.5 * mu * (I_C - 3) - mu * jnp.log(J) + 0.5 * lambda_ * (jnp.log(J) ** 2)
    return W


def corotated(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike] = {},
    cell_data: Mapping[str, jxt.ArrayLike] = {},
    field_data: Mapping[str, jxt.ArrayLike] = {},
) -> jax.Array:
    _: Any
    F: jax.Array = tetra.deformation_gradient(disp, points)
    R: jax.Array
    _, R = jax.scipy.linalg.polar(jax.lax.stop_gradient(F))
    W: jax.Array = mu * jnp.sum((F - R) ** 2) + lambda_ * (jnp.linalg.det(F) - 1) ** 2
    return W


def snh(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike] = {},
    cell_data: Mapping[str, jxt.ArrayLike] = {},
    field_data: Mapping[str, jxt.ArrayLike] = {},
) -> jax.Array:
    F: jax.Array = tetra.deformation_gradient(disp, points)
    W: jax.Array = (
        0.5 * mu * (jnp.sum(F**2) - 3)
        + 0.5 * lambda_ * (jnp.linalg.det(F) - 1 - 3 * mu / 4 * lambda_) ** 2
        - 0.5 * mu * jnp.log(1 + jnp.sum(F**2))
    )
    return W


linear_jac = jax.jacobian(linear)
neo_hookean_jac = jax.jacobian(neo_hookean)
corotated_jac = jax.jacobian(corotated)
nh_jac = jax.jacobian(nh)
# snh_jac = jax.jacobian(snh)


def main() -> None:
    points = np.asarray(
        [
            [1 / np.sqrt(3), 0, 0],
            [-0.5 / np.sqrt(3), 0, 0.5],
            [-0.5 / np.sqrt(3), 0, -0.5],
            [0, np.sqrt(2 / 3), 0],
        ]
    )
    disp = np.zeros((4, 3))
    X = jnp.linspace(-0.5 * np.sqrt(2 / 3), np.sqrt(2 / 3))
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    for x in X:
        disp[3, 1] = x
        y1 = linear_jac(disp, points)[3, 1]
        Y1.append(y1)
        y2 = neo_hookean_jac(disp, points)[3, 1]
        Y2.append(y2)
        y3 = corotated_jac(disp, points)[3, 1]
        Y3.append(y3)
        y4 = nh_jac(disp, points)[3, 1]
        Y4.append(y4)
    plt.figure()
    plt.plot(X, Y1, label="Linear")
    plt.plot(X, Y2, label="Neo-Hookean")
    plt.plot(X, Y3, label="Corotated")
    plt.plot(X, Y4, label="Neo-Hookean 2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
