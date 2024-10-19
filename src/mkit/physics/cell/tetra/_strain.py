import jax.numpy as jnp

import mkit.typing.jax as tj
from mkit.physics.cell import tetra


def deformation_gradient(disp: tj.F43Like, points: tj.F43Like) -> tj.F33:
    u: tj.F43 = jnp.asarray(disp)
    grad_op: tj.F34 = tetra.grad_op(points)
    grad_u: tj.F33 = grad_op @ u
    F: tj.F33 = grad_u + jnp.eye(3)
    return F


def lagrangian_strain(disp: tj.F43Like, points: tj.F43Like) -> tj.F33:
    """Lagrangian finite strain tensor.

    Reference:
        1. <https://en.wikipedia.org/wiki/Finite_strain_theory#Finite_strain_tensors>
    """
    u: tj.F43 = jnp.asarray(disp)
    grad_op: tj.F34 = tetra.grad_op(points)
    grad_u: tj.F33 = grad_op @ u
    E: tj.F33 = 0.5 * (grad_u.T + grad_u + grad_u.T @ grad_u)
    return E


def cauchy_strain(F: tj.F33Like) -> tj.F33:
    """Cauchy strain tensor (right Cauchy-Green deformation tensor).

    Reference:
        1. <https://en.wikipedia.org/wiki/Finite_strain_theory#Cauchy_strain_tensor_(right_Cauchy%E2%80%93Green_deformation_tensor)>
    """
    F: tj.F33 = jnp.asarray(F)
    C: tj.F33 = F.T @ F
    return C
