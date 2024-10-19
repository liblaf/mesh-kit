from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mkit.typing as mt


class ConverterBase(abc.ABC):
    priority: int = 0

    @abc.abstractmethod
    def match_from(self, from_: Any) -> bool: ...

    @abc.abstractmethod
    def match_to(self, to: type) -> bool: ...

    @abc.abstractmethod
    def convert(
        self,
        from_: Any,
        to: type,
        point_data: mt.AttrsLike | None = None,
        cell_data: mt.AttrsLike | None = None,
        field_data: mt.AttrsLike | None = None,
    ) -> Any: ...
