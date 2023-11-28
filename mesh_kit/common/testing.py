from collections.abc import Sequence

from numpy import typing as npt


def assert_shape(arr: npt.ArrayLike, shape: Sequence) -> None:
    assert len(arr.shape) == len(shape), (arr.shape, shape)
    for actual, expected in zip(arr.shape, shape):
        if expected > 0:
            assert actual == expected, (arr.shape, shape)
