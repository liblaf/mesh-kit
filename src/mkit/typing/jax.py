import jax
from jaxtyping import Bool, Float, Integer, Shaped

from mkit.typing.array import (
    ArrayLike,
    BLike,
    BNLike,
    F3Like,
    F4Like,
    F33Like,
    F34Like,
    F43Like,
    F44Like,
    FLike,
    FMN3Like,
    FMNLike,
    FN3Like,
    FNLike,
    FNNLike,
    I2Like,
    I3Like,
    I4Like,
    ILike,
    IN2Like,
    IN3Like,
    IN4Like,
    INLike,
)

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

__all__ = [
    "BN",
    "F3",
    "F4",
    "F33",
    "F34",
    "F43",
    "F44",
    "FMN",
    "FMN3",
    "FN",
    "FN3",
    "FNN",
    "I2",
    "I3",
    "I4",
    "IN",
    "IN2",
    "IN3",
    "IN4",
    "ArrayLike",
    "B",
    "BLike",
    "BNLike",
    "Bool",
    "F",
    "F3Like",
    "F4Like",
    "F33Like",
    "F34Like",
    "F43Like",
    "F44Like",
    "FLike",
    "FMN3Like",
    "FMNLike",
    "FN3Like",
    "FNLike",
    "FNNLike",
    "Float",
    "I",
    "I2Like",
    "I3Like",
    "I4Like",
    "ILike",
    "IN2Like",
    "IN3Like",
    "IN4Like",
    "INLike",
    "Integer",
    "Shaped",
]
