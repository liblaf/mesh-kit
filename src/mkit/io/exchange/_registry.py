from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import sortedcontainers

import mkit.io.exchange as mie

if TYPE_CHECKING:
    import mkit.typing as mt

_C = TypeVar("_C", bound=mie.ConverterBase)
_T = TypeVar("_T")


class ConverterRegistry:
    converters: sortedcontainers.SortedKeyList

    def __init__(self) -> None:
        self.converters = sortedcontainers.SortedKeyList(key=lambda c: c.priority)

    def register(self, converter: _C) -> _C:
        self.converters.add(converter)
        return converter

    def convert(
        self,
        from_: Any,
        to: type[_T],
        point_data: mt.AttrsLike | None = None,
        cell_data: mt.AttrsLike | None = None,
        field_data: mt.AttrsLike | None = None,
    ) -> _T:
        for converter in self.converters:
            assert isinstance(converter, mie.ConverterBase)
            if converter.match_from(from_) and converter.match_to(to):
                return converter.convert(from_, to, point_data, cell_data, field_data)
        raise mie.UnsupportedConversionError(from_, to)


REGISTRY: ConverterRegistry = ConverterRegistry()
