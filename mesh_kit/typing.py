from collections.abc import Callable, Sequence
from typing import Any, ParamSpec, TypeVar

import typeguard

P = ParamSpec("P")
T = TypeVar("T")


def check_type(value: Any, expected_type: type[T]) -> T:
    if __debug__:
        return typeguard.check_type(value, expected_type)
    return value


def typechecked(target: Callable[P, T]) -> Callable[P, T]:
    if __debug__:
        return typeguard.typechecked(target)
    return target


def check_shape(arr: T, expected: tuple[int, ...]) -> T:
    actual: Sequence[int] = check_type(arr.shape, Sequence[int])  # type: ignore
    if __debug__:
        for a, e in zip(actual, expected):
            if e > 0:
                assert a == e, (actual, expected)
    return arr
