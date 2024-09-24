from collections.abc import Callable
from typing import Any, TypeVar

import mkit
import mkit.io._typing as _t

_REGISTRY: dict[tuple[str, str], Callable] = {}
_F = TypeVar("_F", bound=Callable)
_T = TypeVar("_T")


def register(from_: str, to: str) -> Callable[[_F], _F]:
    def decorator(fn: _F) -> _F:
        _REGISTRY[(from_, to)] = fn
        return fn

    return decorator


def convert(from_: Any, to: type[_T], *args, **kwargs) -> _T:
    if isinstance(from_, to):
        return from_
    to_parts: str = mkit.typing.full_name_parts(to)
    for from_type in type(from_).mro():
        from_parts: str = mkit.typing.full_name_parts(from_type)
        for (f, t), fn in _REGISTRY.items():
            if (
                mkit.typing.is_instance_named_partial(from_, f)
                and mkit.typing.is_instance_named_partial
            ):
                return fn(from_, *args, **kwargs)
    raise _t.UnsupportedConversionError(from_, to)
