import jax
import jax.numpy as jnp
import mkit.physics.cell.tetra
import numpy as np


def test_deformation_gradient() -> None:
    key: jax.Array = jax.random.key(0)
    disp: jax.Array = jnp.zeros((4, 3))
    points: jax.Array = jax.random.uniform(key, (4, 3))
    deformation_gradient: jax.Array = mkit.physics.cell.tetra.deformation_gradient(
        disp, points
    )
    np.testing.assert_allclose(deformation_gradient, jnp.eye(3), rtol=0, atol=0)
