import jax.typing as jxt
import numpy.typing as npt
from jaxtyping import Bool, Float, Integer

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
