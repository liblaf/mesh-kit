import jax
import jax.numpy as jnp
import numpy as np

from mkit.physics.cell import tetra


def test_deformation_gradient() -> None:
    key: jax.Array = jax.random.key(0)
    disp: jax.Array = jnp.zeros((4, 3))
    points: jax.Array = jax.random.uniform(key, (4, 3))
    F: jax.Array = tetra.deformation_gradient(disp, points)
    np.testing.assert_allclose(F, jnp.eye(3), rtol=0, atol=0)
