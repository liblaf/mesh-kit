from __future__ import annotations

import bisect
from typing import Any, TypeVar

import mkit.io.exchange as mie
import mkit.typing as mt

_T = TypeVar("_T")


class Registry:
    converters: list[mie.ConverterBase]

    def __init__(self) -> None:
        self.converters = []

    def register(self, converter: mie.ConverterBase) -> None:
        # The number of converters is generally limited. Although `bisect.insert()` has a slow O(n) insertion step, I do not want to introduce additional dependencies (e.g. `sortedcontainers`) or complex code to ensure O(log n) complexity.
        bisect.insort(self.converters, converter, key=lambda c: c.priority)

    def convert(
        self,
        from_: Any,
        to: type[_T],
        *,
        point_data: mt.AttrsLike | None = None,
        cell_data: mt.AttrsLike | None = None,
        field_data: mt.AttrsLike | None = None,
    ) -> _T:
        if isinstance(from_, to):
            return from_
        for converter in self.converters:
            if converter.match_from(from_) and converter.match_to(to):
                return converter.convert(
                    from_,
                    point_data=point_data,
                    cell_data=cell_data,
                    field_data=field_data,
                )
        raise mie.UnsupportedConversionError(from_, to)


CONVERTERS: Registry = Registry()


convert = CONVERTERS.convert
register = CONVERTERS.register
