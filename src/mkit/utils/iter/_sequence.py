from collections.abc import Iterable, Sequence
from typing import Any, TypeGuard, TypeVar

_T = TypeVar("_T")


def flatten(
    iterable: _T | Iterable[_T | Iterable], base_type: tuple[type, ...] = (str, type)
) -> Iterable[_T]:
    def is_iterable(obj: _T | Iterable) -> TypeGuard[Iterable]:
        return isinstance(obj, Iterable) and not isinstance(obj, base_type)

    if not is_iterable(iterable):
        yield iterable  # pyright: ignore [reportReturnType]
        return

    def flatten_iter(iterable: Iterable) -> Iterable[_T]:
        for item in iterable:
            if is_iterable(item):
                yield from flatten_iter(item)
            else:
                yield item

    return flatten_iter(iterable)


def is_subsequence(a: Sequence[Any], b: Sequence[Any]) -> bool:
    i: int = 0
    j: int = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
        j += 1
    return i == len(a)
