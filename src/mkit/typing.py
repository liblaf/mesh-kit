from typing import Any, TypeVar

import typeguard

T = TypeVar("T")


def check_type(val: Any, typ: type[T]) -> T:
    if __debug__:
        return typeguard.check_type(val, typ)
    else:
        return val
