import jax.numpy as jnp

import mkit.typing.jax as j
from mkit.physics.cell import tetra


def deformation_gradient(disp: j.D43Like, points: j.D43Like) -> j.D33:
    u: j.D43 = jnp.asarray(disp)
    grad_op: j.D34 = tetra.grad_op(points)
    grad_u: j.D33 = grad_op @ u
    F: j.D33 = grad_u + jnp.eye(3)
    return F


def lagrangian_strain(disp: j.D43Like, points: j.D43Like) -> j.D33:
    """Lagrangian finite strain tensor.

    Reference:
        1. <https://en.wikipedia.org/wiki/Finite_strain_theory#Finite_strain_tensors>
    """
    u: j.D43 = jnp.asarray(disp)
    grad_op: j.D34 = tetra.grad_op(points)
    grad_u: j.D33 = grad_op @ u
    E: j.D33 = 0.5 * (grad_u.T + grad_u + grad_u.T @ grad_u)
    return E


def cauchy_strain(_F: j.D33Like) -> j.D33:
    """Cauchy strain tensor (right Cauchy-Green deformation tensor).

    Reference:
        1. <https://en.wikipedia.org/wiki/Finite_strain_theory#Cauchy_strain_tensor_(right_Cauchy%E2%80%93Green_deformation_tensor)>
    """
    F: j.D33 = jnp.asarray(_F)
    C: j.D33 = F.T @ F
    return C
