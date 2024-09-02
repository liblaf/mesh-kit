from collections.abc import Sequence
from typing import Any

from mkit.typing import fullname


def is_instance_named(obj: Any, name: str | Sequence[str]) -> bool:
    if isinstance(name, str):
        name = [name]
    return any(fullname(clazz) in name for clazz in type(obj).mro())
