from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import jax.typing as jxt

import mkit.math as m
from mkit.physics.cell import tetra
from mkit.physics.energy.abc import cell_energy


@cell_energy
def linear(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])
    mu: jax.Array = jnp.asarray(cell_data["mu"])
    grad_op: jax.Array = tetra.grad_op(points)  # (3, 4)
    grad_u: jax.Array = grad_op @ disp  # (3, 3)
    E: jax.Array = 0.5 * (grad_u.T + grad_u)  # (3, 3)
    W: jax.Array = 0.5 * lambda_ * jnp.trace(E) ** 2 + mu * jnp.trace(E @ E)
    return W


@cell_energy
def saint_venant_kirchhoff_wiki(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Saint Venant-Kirchhoff.

    Refs:
        1. <https://en.wikipedia.org/wiki/Hyperelastic_material>
    """
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])  # Lamé's first parameter
    mu: jax.Array = jnp.asarray(cell_data["mu"])  # Lamé's second parameter
    grad_op: jax.Array = tetra.grad_op(points)  # (3, 4)
    grad_u: jax.Array = grad_op @ disp  # (3, 3)
    E: jax.Array = 0.5 * (
        grad_u.T + grad_u + grad_u.T @ grad_u
    )  # (3, 3) Lagrangian Green strain
    W: jax.Array = 0.5 * lambda_ * jnp.trace(E) ** 2 + mu * jnp.trace(E @ E)
    return W


@cell_energy
def neo_hookean_wiki(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Neo-Hookean.

    For consistency with linear elasticity,
    C1 = mu / 2; D1 = lambda_L / 2
    where `lambda_L` is the first Lamé parameter and `mu` is the shear modulus or the second Lamé parameter.

    Refs:
        1. <https://en.wikipedia.org/wiki/Neo-Hookean_solid>
    """
    C1: jax.Array = jnp.asarray(cell_data["C1"])
    D1: jax.Array = jnp.asarray(cell_data["D1"])
    F: jax.Array = tetra.deformation_gradient(disp, points)
    C: jax.Array = F.T @ F  # right Cauchy-Green deformation tensor
    I1: jax.Array = jnp.trace(C)
    J: jax.Array = jnp.linalg.det(F)
    lnJ: jax.Array = jnp.log(J)
    W: jax.Array = C1 * (I1 - 3 - 2 * lnJ) + D1 * (J - 1) ** 2
    return W


@cell_energy
def neo_hookean_wiki_alternative(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Neo-Hookean.

    For consistency with linear elasticity,
    C1 = mu / 2; D1 = lambda_L / 2
    where `lambda_L` is the first Lamé parameter and `mu` is the shear modulus or the second Lamé parameter.

    Refs:
        1. <https://en.wikipedia.org/wiki/Neo-Hookean_solid>
    """
    C1: jax.Array = jnp.asarray(cell_data["C1"])
    D1: jax.Array = jnp.asarray(cell_data["D1"])
    F: jax.Array = tetra.deformation_gradient(disp, points)
    C: jax.Array = F.T @ F  # right Cauchy-Green deformation tensor
    I1: jax.Array = jnp.trace(C)
    J: jax.Array = jnp.linalg.det(F)
    I1bar: jax.Array = J ** (-2 / 3) * I1
    W: jax.Array = C1 * (I1bar - 3) + (C1 / 6 + D1 / 4) * (J**2 + 1 / J**2 - 2)
    return W


@cell_energy
def mooney_rivlin_wiki(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Mooney-Rivlin.

    For consistency with linear elasticity in the limit of small strains, it is necessary that
    kappa = 2 / D1; mu = 2 * (C01 + C10)
    where `kappa` is the bulk modulus and `mu` is the shear modulus.

    Refs:
        1. <https://en.wikipedia.org/wiki/Mooney%E2%80%93Rivlin_solid>
    """
    C1: jax.Array = jnp.asarray(cell_data["C1"])
    C2: jax.Array = jnp.asarray(cell_data["C2"])
    D1: jax.Array = jnp.asarray(cell_data["D1"])
    C01: jax.Array = C2
    C10: jax.Array = C1
    F: jax.Array = tetra.deformation_gradient(disp, points)
    B: jax.Array = F @ F.T  # left Cauchy-Green deformation tensor
    B_bar: jax.Array = jnp.linalg.det(B) ** (-1 / 3) * B
    I1bar: jax.Array = jnp.trace(B_bar)
    I2bar: jax.Array = 0.5 * (jnp.trace(B) ** 2 - jnp.trace(B @ B))
    J: jax.Array = jnp.linalg.det(F)
    W: jax.Array = C01 * (I2bar - 3) + C10 * (I1bar - 3) + 1 / D1 * (J - 1) ** 2
    return W


@cell_energy
def yeoh_wiki(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Yeoh.

    Refs:
        1. <https://en.wikipedia.org/wiki/Yeoh_hyperelastic_model>
    """
    C10: jax.Array = cell_data["C10"]
    C20: jax.Array = cell_data["C20"]
    C30: jax.Array = cell_data["C30"]
    C11: jax.Array = jnp.asarray(cell_data["C11"])
    F: jax.Array = tetra.deformation_gradient(disp, points)
    B: jax.Array = F @ F.T  # left Cauchy-Green deformation tensor
    B_bar: jax.Array = jnp.linalg.det(B) ** (-1 / 3) * B
    I1bar: jax.Array = jnp.trace(B_bar)
    J: jax.Array = jnp.linalg.det(F)
    W: jax.Array = (
        C10 * (I1bar - 3)
        + C20 * (I1bar - 3) ** 2
        + C30 * (I1bar - 3) ** 3
        + C11 * (J - 1) ** 2
    )
    return W


@cell_energy
def corotated_mcadams(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Corotational [McAdams 2011].

    Refs:
        1. McAdams, Aleka, et al. "Efficient elasticity for character skinning with contact and collisions." ACM SIGGRAPH 2011 papers. 2011. 1-12.
        2. Smith, Breannan, Fernando De Goes, and Theodore Kim. "Stable neo-hookean flesh simulation." ACM Transactions on Graphics (TOG) 37.2 (2018): 1-15.
    """
    _: Any
    mu: jax.Array = jnp.asarray(cell_data["mu"])  # Lamé's second parameter
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])  # Lamé's first parameter
    F: jax.Array = tetra.deformation_gradient(disp, points)
    R: jax.Array
    _, R = m.polar(F)
    S: jax.Array = R.T @ F
    I: jax.Array = jnp.eye(3)  # noqa: E741
    W: jax.Array = mu * m.F2(F - R) + 0.5 * lambda_ * jnp.trace(S - I) ** 2
    return W


@cell_energy
def corotated_stomakhin(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Corotated [Stomakhin 2012].

    Refs:
        1. Stomakhin, Alexey, et al. "Energetically Consistent Invertible Elasticity." Symposium on Computer Animation. Vol. 1. No. 2. 2012.
        2. Chen, Yizhou, et al. "Position-Based Nonlinear Gauss-Seidel for Quasistatic Hyperelasticity." arXiv preprint arXiv:2306.09021 (2023).
    """
    _: Any
    mu: jax.Array = jnp.asarray(cell_data["mu"])
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])
    F: jax.Array = tetra.deformation_gradient(disp, points)
    R: jax.Array
    _, R = m.polar(F)
    J: jax.Array = jnp.linalg.det(F)  # relative volume change
    W: jax.Array = mu * m.F2(F - R) + 0.5 * lambda_ * (J - 1) ** 2
    return W


corotated = corotated_mcadams


@cell_energy
def neo_hookean_bonet(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Neo-Hookean [Bonet 1997].

    Refs:
        1. Bonet, Javier, and Richard D. Wood. Nonlinear continuum mechanics for finite element analysis. Cambridge university press, 1997.
        2. Smith, Breannan, Fernando De Goes, and Theodore Kim. "Stable neo-hookean flesh simulation." ACM Transactions on Graphics (TOG) 37.2 (2018): 1-15.
    """
    mu: jax.Array = jnp.asarray(cell_data["mu"])  # Lamé's second parameter
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])  # Lamé's first parameter
    F: jax.Array = tetra.deformation_gradient(disp, points)
    J: jax.Array = jnp.linalg.det(F)  # relative volume change
    C: jax.Array = F.T @ F  # right Cauchy-Green tensor
    I_C: jax.Array = jnp.trace(C)
    logJ: jax.Array = jnp.log(J)
    W: jax.Array = 0.5 * mu * (I_C - 3) - mu * logJ + 0.5 * lambda_ * logJ**2
    return W


@cell_energy
def neo_hookean_ogden(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Neo-Hookean [Ogden 1997].

    Refs:
        1. Ogden, Raymond W. Non-linear elastic deformations. Courier Corporation, 1997.
        2. Smith, Breannan, Fernando De Goes, and Theodore Kim. "Stable neo-hookean flesh simulation." ACM Transactions on Graphics (TOG) 37.2 (2018): 1-15.
    """
    mu: jax.Array = jnp.asarray(cell_data["mu"])  # Lamé's second parameter
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])  # Lamé's first parameter
    F: jax.Array = tetra.deformation_gradient(disp, points)
    J: jax.Array = jnp.linalg.det(F)  # relative volume change
    C: jax.Array = F.T @ F  # right Cauchy-Green tensor
    I_C: jax.Array = jnp.trace(C)
    logJ: jax.Array = jnp.log(J)
    W: jax.Array = 0.5 * mu * (I_C - 3) - mu * logJ + 0.5 * lambda_ * (J - 1) ** 2
    return W


@cell_energy
def neo_hookean_bower(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Neo-Hookean [Bower 2009].

    Refs:
        1. Bower, Allan F. Applied mechanics of solids. CRC press, 2009.
        2. Smith, Breannan, Fernando De Goes, and Theodore Kim. "Stable neo-hookean flesh simulation." ACM Transactions on Graphics (TOG) 37.2 (2018): 1-15.
    """
    mu: jax.Array = jnp.asarray(cell_data["mu"])  # Lamé's second parameter
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])  # Lamé's first parameter
    F: jax.Array = tetra.deformation_gradient(disp, points)
    J: jax.Array = jnp.linalg.det(F)  # relative volume change
    C: jax.Array = F.T @ F  # right Cauchy-Green tensor
    I_C: jax.Array = jnp.trace(C)
    W: jax.Array = 0.5 * mu * (J ** (-2 / 3) * I_C - 3) + 0.5 * lambda_ * (J - 1) ** 2
    return W


@cell_energy
def neo_hookean_wang(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Neo-Hookean [Wang 2016].

    Refs:
        1. Wang, Huamin, and Yin Yang. "Descent methods for elastic body simulation on the GPU." ACM Transactions on Graphics (TOG) 35.6 (2016): 1-10.
        2. Smith, Breannan, Fernando De Goes, and Theodore Kim. "Stable neo-hookean flesh simulation." ACM Transactions on Graphics (TOG) 37.2 (2018): 1-15.
    """
    mu: jax.Array = jnp.asarray(cell_data["mu"])  # Lamé's second parameter
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])  # Lamé's first parameter
    F: jax.Array = tetra.deformation_gradient(disp, points)
    J: jax.Array = jnp.linalg.det(F)  # relative volume change
    C: jax.Array = F.T @ F  # right Cauchy-Green tensor
    I_C: jax.Array = jnp.trace(C)
    W: jax.Array = 0.5 * mu * (J ** (-2 / 3) * I_C - 3) + 0.5 * lambda_ * (J - 1)
    return W


@cell_energy
def neo_hookean_macklin(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Neo-Hookean [Macklin 2021].

    Refs:
        1. Macklin, Miles, and Matthias Muller. "A constraint-based formulation of stable neo-hookean materials." Proceedings of the 14th ACM SIGGRAPH conference on motion, interaction and games. 2021.
        2. Chen, Yizhou, et al. "Position-Based Nonlinear Gauss-Seidel for Quasistatic Hyperelasticity." arXiv preprint arXiv:2306.09021 (2023).
    """
    mu: jax.Array = jnp.asarray(cell_data["mu"])  # Lamé's second parameter
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])  # Lamé's first parameter
    F: jax.Array = tetra.deformation_gradient(disp, points)
    J: jax.Array = jnp.linalg.det(F)  # relative volume change
    lambda_hat: jax.Array = mu + lambda_
    W: jax.Array = (
        0.5 * mu * m.F2(F) + 0.5 * lambda_hat * (J - 1 - mu / lambda_hat) ** 2
    )
    return W


@cell_energy
def neo_hookean_stable_smith(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Stable Neo-Hookean [Smith 2018].

    Refs:
        1. Smith, Breannan, Fernando De Goes, and Theodore Kim. "Stable neo-hookean flesh simulation." ACM Transactions on Graphics (TOG) 37.2 (2018): 1-15.
        2. Chen, Yizhou, et al. "Position-Based Nonlinear Gauss-Seidel for Quasistatic Hyperelasticity." arXiv preprint arXiv:2306.09021 (2023).
    """
    mu: jax.Array = jnp.asarray(cell_data["mu"])  # Lamé's second parameter
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])  # Lamé's first parameter
    F: jax.Array = tetra.deformation_gradient(disp, points)
    J: jax.Array = jnp.linalg.det(F)  # relative volume change
    W: jax.Array = (
        0.5 * mu * (m.F2(F) - 3)
        + 0.5 * lambda_ * (J - 1 - 3 * mu / (4 * lambda_)) ** 2
        - 0.5 * mu * jnp.log(1 + m.F2(F))
    )
    return W


neo_hookean = neo_hookean_macklin


@cell_energy
def st_venant_kirchhoff_wang(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """St. Venant-Kirchhoff.

    Refs:
        1. Wang, Huamin, and Yin Yang. "Descent methods for elastic body simulation on the GPU." ACM Transactions on Graphics (TOG) 35.6 (2016): 1-10.
    """
    mu: jax.Array = jnp.asarray(cell_data["mu"])  # Lamé's second parameter
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])  # Lamé's first parameter
    s0: jax.Array = mu  # shear modulus
    s1: jax.Array = lambda_ + 2 / 3 * mu  # bulk modulus
    F: jax.Array = tetra.deformation_gradient(disp, points)
    C: jax.Array = F.T @ F  # right Cauchy-Green deformation tensor
    I: jax.Array = jnp.trace(C)  # 1st invariant  # noqa: E741
    II: jax.Array = jnp.trace(C @ C)  # 2nd invariant
    W: jax.Array = 0.5 * s0 * (I - 3) ** 2 + 0.25 * s1 * (II - 2 * I + 3)
    return W


@cell_energy
def neo_hookean_ogden_wang(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Neo-Hookean [Ogden 1997].

    Refs:
        1. Ogden, Raymond W. Non-linear elastic deformations. Courier Corporation, 1997.
        2. Wang, Huamin, and Yin Yang. "Descent methods for elastic body simulation on the GPU." ACM Transactions on Graphics (TOG) 35.6 (2016): 1-10.
    """
    mu: jax.Array = jnp.asarray(cell_data["mu"])  # Lamé's second parameter
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])  # Lamé's first parameter
    s0: jax.Array = mu  # shear modulus
    s1: jax.Array = lambda_ + 2 / 3 * mu  # bulk modulus
    F: jax.Array = tetra.deformation_gradient(disp, points)
    C: jax.Array = F.T @ F  # right Cauchy-Green deformation tensor
    I: jax.Array = jnp.trace(C)  # 1st invariant  # noqa: E741
    III: jax.Array = jnp.linalg.det(C)  # 3rd invariant
    W: jax.Array = s0 * (III ** (-1 / 3) * I - 3) + s1 * (III**-0.5 - 1)
    return W


@cell_energy
def mooney_rivlin_wang(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Mooney-Rivlin [Macosko 1994].

    Refs:
        1. Macosko, C. W. "Rheology principles, measurements, and applications, VCH Publ." Inc, New York (1994).
        2. Wang, Huamin, and Yin Yang. "Descent methods for elastic body simulation on the GPU." ACM Transactions on Graphics (TOG) 35.6 (2016): 1-10.
    """
    mu: jax.Array = jnp.asarray(cell_data["mu"])  # Lamé's second parameter
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])  # Lamé's first parameter
    s0: jax.Array = mu  # shear modulus
    s1: jax.Array = lambda_ + 2 / 3 * mu  # bulk modulus
    F: jax.Array = tetra.deformation_gradient(disp, points)
    C: jax.Array = F.T @ F  # right Cauchy-Green deformation tensor
    I: jax.Array = jnp.trace(C)  # 1st invariant  # noqa: E741
    III: jax.Array = jnp.linalg.det(C)  # 3rd invariant
    raise NotImplementedError
    W: jax.Array = s0 * (III ** (-1 / 3) * I - 3) + s1 * (III**-0.5 - 1)
    return W


@cell_energy
def fung(
    disp: jxt.ArrayLike,
    points: jxt.ArrayLike,
    point_data: Mapping[str, jxt.ArrayLike],
    cell_data: Mapping[str, jxt.ArrayLike],
    field_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    """Fung [Fung 1993].

    Refs:
        1. Fung, Yuan-cheng. Biomechanics: mechanical properties of living tissues. Springer Science & Business Media, 2013.
        2. Wang, Huamin, and Yin Yang. "Descent methods for elastic body simulation on the GPU." ACM Transactions on Graphics (TOG) 35.6 (2016): 1-10.
    """
    mu: jax.Array = jnp.asarray(cell_data["mu"])  # Lamé's second parameter
    lambda_: jax.Array = jnp.asarray(cell_data["lambda"])  # Lamé's first parameter
    s0: jax.Array = mu  # shear modulus
    s1: jax.Array = lambda_ + 2 / 3 * mu  # bulk modulus
    F: jax.Array = tetra.deformation_gradient(disp, points)
    C: jax.Array = F.T @ F  # right Cauchy-Green deformation tensor
    I: jax.Array = jnp.trace(C)  # 1st invariant  # noqa: E741
    III: jax.Array = jnp.linalg.det(C)  # 3rd invariant
    raise NotImplementedError
    W: jax.Array = s0 * (III ** (-1 / 3) * I - 3) + s1 * (III**-0.5 - 1)
    return W
