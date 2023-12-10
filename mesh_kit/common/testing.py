from collections.abc import Sequence

from numpy import typing as npt

__all__ = ["assert_shape"]


def assert_shape(arr: npt.NDArray, shape: Sequence) -> None:
    assert len(arr.shape) == len(shape), (arr.shape, shape)
    for actual, expected in zip(arr.shape, shape):
        if expected > 0:
            assert actual == expected, (arr.shape, shape)
