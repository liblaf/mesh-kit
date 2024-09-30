import abc
from typing import Any

from mkit.typing import AttributeArray, AttributesLike


class C2CMethod(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, source: Any, target: Any, data: AttributesLike | None = None
    ) -> dict[str, AttributeArray]: ...


class P2PMethod(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, source: Any, target: Any, data: AttributesLike | None = None
    ) -> dict[str, AttributeArray]: ...
