import jax
from jaxtyping import Bool, Float, Integer

B = Bool[jax.Array, ""]
BN = Bool[jax.Array, "N"]

F = Float[jax.Array, ""]
F3 = Float[jax.Array, "3"]
F33 = Float[jax.Array, "3 3"]
F34 = Float[jax.Array, "3 4"]
F4 = Float[jax.Array, "4"]
F43 = Float[jax.Array, "4 3"]
F44 = Float[jax.Array, "4 4"]
FMN = Float[jax.Array, "M N"]
FMN3 = Float[jax.Array, "M N 3"]
FN = Float[jax.Array, "N"]
FN3 = Float[jax.Array, "N 3"]
FNN = Float[jax.Array, "N N"]

I = Integer[jax.Array, ""]  # noqa: E741
I2 = Integer[jax.Array, "2"]
I3 = Integer[jax.Array, "3"]
I4 = Integer[jax.Array, "4"]
IN = Integer[jax.Array, "N"]
IN2 = Integer[jax.Array, "N 2"]
IN3 = Integer[jax.Array, "N 3"]
IN4 = Integer[jax.Array, "N 4"]
