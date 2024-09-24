from typing import Any, Protocol

from mkit.typing import AttributeArray, AttributesLike


class TransferFn(Protocol):
    def __call__(
        self,
        source: Any,
        target: Any,
        data: AttributesLike | None = None,
        *,
        distance_threshold: float = 0.1,
    ) -> dict[str, AttributeArray]: ...
