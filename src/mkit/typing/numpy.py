import numpy as np
from jaxtyping import Bool, Float, Integer, Shaped

from mkit.typing import is_array_like
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

B = Bool[np.ndarray, ""]
BN = Bool[np.ndarray, "N"]

F = Float[np.ndarray, ""]
F3 = Float[np.ndarray, "3"]
F33 = Float[np.ndarray, "3 3"]
F34 = Float[np.ndarray, "3 4"]
F4 = Float[np.ndarray, "4"]
F43 = Float[np.ndarray, "4 3"]
F44 = Float[np.ndarray, "4 4"]
FMN = Float[np.ndarray, "M N"]
FMN3 = Float[np.ndarray, "M N 3"]
FN = Float[np.ndarray, "N"]
FN3 = Float[np.ndarray, "N 3"]
FNN = Float[np.ndarray, "N N"]

I = Integer[np.ndarray, ""]  # noqa: E741
I2 = Integer[np.ndarray, "2"]
I3 = Integer[np.ndarray, "3"]
I4 = Integer[np.ndarray, "4"]
IN = Integer[np.ndarray, "N"]
IN2 = Integer[np.ndarray, "N 2"]
IN3 = Integer[np.ndarray, "N 3"]
IN4 = Integer[np.ndarray, "N 4"]

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
    "is_array_like",
]
