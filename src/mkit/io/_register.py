import bisect
from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar

import mkit.io._typing as _t

_F = TypeVar("_F", bound=Callable)
_T = TypeVar("_T")


class RegistryItem(NamedTuple):
    priority: int
    from_: str
    to: str
    fn: Callable


class Registry:
    converters: list[RegistryItem]

    def __init__(self) -> None:
        self.converters = []

    def register(self, from_: str, to: str, priority: int = 0) -> Callable[[_F], _F]:
        def decorator(fn: _F) -> _F:
            item: RegistryItem = RegistryItem(priority, from_, to, fn)
            self.converters.insert(
                bisect.bisect_right(
                    self.converters, item.priority, key=lambda item: item.priority
                ),
                item,
            )
            return fn

        return decorator

    def convert(self, from_: Any, to: type[_T], *args, **kwargs) -> _T:
        if isinstance(from_, to):
            return from_
        for item in self.converters:
            if _t.is_sub_type(from_, item.from_) and _t.is_sub_type(to, item.to):
                return item.fn(from_, *args, **kwargs)
        raise _t.UnsupportedConversionError(from_, to)


REGISTRY = Registry()


def convert(from_: Any, to: type[_T], *args, **kwargs) -> _T:
    return REGISTRY.convert(from_, to, *args, **kwargs)
