from typing import Protocol, TypeVar, cast, runtime_checkable

import typeguard

_T = TypeVar("_T")


def check_type(typ: type[_T], val: object) -> _T:
    if __debug__:
        return typeguard.check_type(val, typ)
    else:
        return cast(typ, val)


@runtime_checkable
class Shaped(Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...


_Shaped = TypeVar("_Shaped", bound=Shaped)


def assert_shape(arr: _Shaped, expected: tuple[int, ...]) -> _Shaped:
    if __debug__:
        assert len(arr.shape) == len(expected), (arr.shape, expected)
        for a, e in zip(arr.shape, expected):
            if e >= 0:
                assert a == e, (arr.shape, expected)
    return arr
