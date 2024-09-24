import torch
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

B = Bool[torch.Tensor, ""]
BN = Bool[torch.Tensor, "N"]

F = Float[torch.Tensor, ""]
F3 = Float[torch.Tensor, "3"]
F33 = Float[torch.Tensor, "3 3"]
F34 = Float[torch.Tensor, "3 4"]
F4 = Float[torch.Tensor, "4"]
F43 = Float[torch.Tensor, "4 3"]
F44 = Float[torch.Tensor, "4 4"]
FMN = Float[torch.Tensor, "M N"]
FMN3 = Float[torch.Tensor, "M N 3"]
FN = Float[torch.Tensor, "N"]
FN3 = Float[torch.Tensor, "N 3"]
FNN = Float[torch.Tensor, "N N"]

I = Integer[torch.Tensor, ""]  # noqa: E741
I2 = Integer[torch.Tensor, "2"]
I3 = Integer[torch.Tensor, "3"]
I4 = Integer[torch.Tensor, "4"]
IN = Integer[torch.Tensor, "N"]
IN2 = Integer[torch.Tensor, "N 2"]
IN3 = Integer[torch.Tensor, "N 3"]
IN4 = Integer[torch.Tensor, "N 4"]

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
