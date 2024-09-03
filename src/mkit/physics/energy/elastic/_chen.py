from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import jax.typing as jxt

import mkit.math.jax as m
from mkit.physics.cell import tetra
from mkit.physics.energy import cell_energy


@cell_energy
def corotated(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Corotated (Stomakhin 2012).

    Reference:
        1. Chen, Yizhou, et al. "Position-Based Nonlinear Gauss-Seidel for Quasistatic Hyperelasticity." arXiv preprint arXiv:2306.09021 (2023).
        2. Stomakhin, Alexey, et al. "Energetically Consistent Invertible Elasticity." Symposium on Computer Animation. Vol. 1. No. 2. 2012.
    """
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])
    mu: jax.Array = jnp.asarray(cell_data["mu"])
    F: jax.Array = tetra.deformation_gradient(disp, points)  # (3, 3)
    J: jax.Array = jnp.linalg.det(F)
    S: jax.Array  # (3, 3)
    R: jax.Array  # (3, 3)
    S, R = m.polar(F)
    W: jax.Array = mu * m.F2(F - R) + 0.5 * lambda_ * (J - 1) ** 2
    return W


@cell_energy
def neo_hookean(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Neo-Hookean (Macklin 2021).

    Reference:
        1. Chen, Yizhou, et al. "Position-Based Nonlinear Gauss-Seidel for Quasistatic Hyperelasticity." arXiv preprint arXiv:2306.09021 (2023).
        2. Macklin, Miles, and Matthias Muller. "A constraint-based formulation of stable neo-hookean materials." Proceedings of the 14th ACM SIGGRAPH conference on motion, interaction and games. 2021.
    """
    _: Any
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])
    mu: jax.Array = jnp.asarray(cell_data["mu"])
    lambda_hat: jax.Array = mu + lambda_
    F: jax.Array = tetra.deformation_gradient(disp, points)  # (3, 3)
    C: jax.Array = tetra.cauchy_strain(F)  # (3, 3)
    J: jax.Array = jnp.linalg.det(F)  # ()
    I1_C: jax.Array  # (3, 3)
    I1_C, _, _ = m.invariants(C)
    W: jax.Array = 0.5 * mu * I1_C + 0.5 * lambda_hat * (J - 1 - mu / lambda_hat) ** 2
    return W


@cell_energy
def stable_neo_hookean(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Stable Neo-Hookean (Smith 2018).

    Reference:
        1. Chen, Yizhou, et al. "Position-Based Nonlinear Gauss-Seidel for Quasistatic Hyperelasticity." arXiv preprint arXiv:2306.09021 (2023).
        2. Smith, Breannan, Fernando De Goes, and Theodore Kim. "Stable neo-hookean flesh simulation." ACM Transactions on Graphics (TOG) 37.2 (2018): 1-15.
    """
    _: Any
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])
    mu: jax.Array = jnp.asarray(cell_data["mu"])
    lambda_hat: jax.Array = mu + lambda_
    F: jax.Array = tetra.deformation_gradient(disp, points)  # (3, 3)
    C: jax.Array = tetra.cauchy_strain(F)  # (3, 3)
    J: jax.Array = jnp.linalg.det(F)  # ()
    I1_C: jax.Array  # (3, 3)
    I1_C, _, _ = m.invariants(C)
    W: jax.Array = 0.5 * mu * I1_C + 0.5 * lambda_hat * (J - 1 - mu / lambda_hat) ** 2
    return W
