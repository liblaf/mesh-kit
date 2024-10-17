from typing import Any


def is_array_like(obj: Any) -> bool:
    return hasattr(obj, "__len__") and not isinstance(obj, str | bytes)
