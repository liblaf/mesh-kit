from __future__ import annotations

from typing import Any


class UnsupportedConversionError(ValueError):
    from_: Any
    to: type

    def __init__(self, from_: Any, to: type) -> None:
        super().__init__(f"Cannot convert from {from_} to {to}")
        self.from_ = from_
        self.to = to
