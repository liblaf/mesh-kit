import pathlib
from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable

import numpy as np
import torch
import typeguard
from numpy import typing as npt

StrPath: TypeAlias = str | pathlib.Path


@runtime_checkable
class Shaped(Protocol):
    shape: Sequence[int]


A = TypeVar("A", npt.NDArray, torch.Tensor)
S = TypeVar("S", bound=Shaped)
T = TypeVar("T")


def check_type(val: Any, typ: type[T]) -> T:
    if __debug__:
        return typeguard.check_type(val, typ)
    else:
        return val


def check_shape(val: S, shape: Sequence[int]) -> S:
    if __debug__:
        for a, b in zip(val.shape, shape):
            if b < 0:
                continue
            assert a == b, (val.shape, shape)
    return val


def check_dtype(val: A, dtype: np.dtype | torch.dtype) -> A:
    if __debug__:
        assert val.dtype == dtype, (val.dtype, dtype)
    return val
