import jax
import jax.numpy as jnp
import mkit.physics.common.element
import numpy as np


def test_deformation_gradient() -> None:
    key: jax.Array = jax.random.key(0)
    points: jax.Array = jax.random.uniform(key, (4, 3))
    deformation_gradient: jax.Array = (
        mkit.physics.common.element.calc_deformation_gradient(points, points)
    )
    np.testing.assert_allclose(deformation_gradient, jnp.eye(3), rtol=1e-15, atol=1e-15)
