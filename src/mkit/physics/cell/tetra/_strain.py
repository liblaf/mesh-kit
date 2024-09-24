import jax.numpy as jnp

import mkit.typing.jax as jt
from mkit.physics.cell import tetra


def deformation_gradient(disp: jt.F43Like, points: jt.F43Like) -> jt.F33:
    u: jt.F43 = jnp.asarray(disp)
    grad_op: jt.F34 = tetra.grad_op(points)
    grad_u: jt.F33 = grad_op @ u
    F: jt.F33 = grad_u + jnp.eye(3)
    return F


def lagrangian_strain(disp: jt.F43Like, points: jt.F43Like) -> jt.F33:
    """Lagrangian finite strain tensor.

    Reference:
        1. <https://en.wikipedia.org/wiki/Finite_strain_theory#Finite_strain_tensors>
    """
    u: jt.F43 = jnp.asarray(disp)
    grad_op: jt.F34 = tetra.grad_op(points)
    grad_u: jt.F33 = grad_op @ u
    E: jt.F33 = 0.5 * (grad_u.T + grad_u + grad_u.T @ grad_u)
    return E


def cauchy_strain(F: jt.F33Like) -> jt.F33:
    """Cauchy strain tensor (right Cauchy-Green deformation tensor).

    Reference:
        1. <https://en.wikipedia.org/wiki/Finite_strain_theory#Cauchy_strain_tensor_(right_Cauchy%E2%80%93Green_deformation_tensor)>
    """
    F: jt.F33 = jnp.asarray(F)
    C: jt.F33 = F.T @ F
    return C
