import os
from collections.abc import Sequence
from typing import Protocol, TypeVar

import jax.numpy as jnp

StrPath = str | os.PathLike


class ArrayLike(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...


T = TypeVar("T", bound=ArrayLike)


def check_shape(arr: T, shape: Sequence[int]) -> T:
    if not hasattr(arr, "shape"):
        arr = jnp.asarray(arr)
    assert len(arr.shape) == len(shape), (arr.shape, shape)
    for a, b in zip(arr.shape, shape, strict=True):
        if b < 0:
            continue
        assert a == b, (arr.shape, shape)
    return arr
