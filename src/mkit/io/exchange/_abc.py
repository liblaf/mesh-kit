from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    import pyvista as pv

    import mkit.typing as mt


class ConverterBase(abc.ABC):
    _priority: int = 0
    _from: type
    _to: type

    @property
    def priority(self) -> int:
        return self._priority

    def match_from(self, from_: Any) -> bool:
        return isinstance(from_, self._from)

    def match_to(self, to: type) -> bool:
        return issubclass(to, self._to)

    @abc.abstractmethod
    def convert(
        self,
        from_: Any,
        *,
        point_data: mt.AttrsLike | None = None,
        cell_data: mt.AttrsLike | None = None,
        field_data: mt.AttrsLike | None = None,
    ) -> Any: ...

    def warn_not_supported_association(
        self, association: pv.FieldAssociation, attr: mt.AttrsLike | None = None
    ) -> None:
        if attr is None:
            return
        logger.warning("{:r} does not support {:r} data", self._to, association)
