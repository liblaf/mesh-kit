from typing import Any

from mkit.ops.transfer import C2CMethod, P2PMethod
from mkit.typing import AttributeArray, AttributesLike


def surface_to_surface(
    source: Any,
    target: Any,
    data: AttributesLike | None = None,
    *,
    method: C2CMethod | P2PMethod,
) -> dict[str, AttributeArray]:
    result: dict[str, AttributeArray] = method(source, target, data)
    return result
