from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

import mkit.typing as mt

_T = TypeVar("_T")


def flatten(
    iterable: _T | Iterable[_T] | Iterable[Iterable[_T]] | Iterable,
    base_type: tuple[type, ...] = (str, bytes),
) -> Iterable[_T]:
    if not mt.is_iterable(iterable, base_type):
        yield iterable  # pyright: ignore [reportReturnType]
        return

    for item in iterable:
        if mt.is_iterable(item, base_type):
            yield from flatten(item)
        else:
            yield item


def is_subsequence(a: Sequence[Any], b: Sequence[Any]) -> bool:
    i: int = 0
    j: int = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
        j += 1
    return i == len(a)
