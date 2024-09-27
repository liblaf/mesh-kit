import jax.typing as jxt
import numpy.typing as npt
from jaxtyping import Bool, Float, Integer, Shaped

from mkit.typing import is_array_like

ArrayLike = npt.ArrayLike | jxt.ArrayLike

BLike = Bool[ArrayLike, ""]
BNLike = Bool[ArrayLike, "N"]

F33Like = Float[ArrayLike, "3 3"]
F34Like = Float[ArrayLike, "3 4"]
F3Like = Float[ArrayLike, "3"]
F43Like = Float[ArrayLike, "4 3"]
F44Like = Float[ArrayLike, "4 4"]
F4Like = Float[ArrayLike, "4"]
FLike = Float[ArrayLike, ""]
FMN3Like = Float[ArrayLike, "M N 3"]
FMNLike = Float[ArrayLike, "M N"]
FN3Like = Float[ArrayLike, "N 3"]
FNLike = Float[ArrayLike, "N"]
FNNLike = Float[ArrayLike, "N N"]

I2Like = Integer[ArrayLike, "2"]
I3Like = Integer[ArrayLike, "3"]
I4Like = Integer[ArrayLike, "4"]
ILike = Integer[ArrayLike, ""]
IN2Like = Integer[ArrayLike, "N 2"]
IN3Like = Integer[ArrayLike, "N 3"]
IN4Like = Integer[ArrayLike, "N 4"]
INLike = Integer[ArrayLike, "N"]

__all__ = [
    "ArrayLike",
    "BLike",
    "BNLike",
    "Bool",
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
    "is_array_like",
]
