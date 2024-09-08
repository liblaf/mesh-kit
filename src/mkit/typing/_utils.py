from collections.abc import Sequence
from typing import Any


def fullname(obj: Any) -> str:
    """Returns the fully qualified name of the given object.

    Args:
        obj: The object whose fully qualified name is to be returned.

    Returns:
        The fully qualified name of the object.

    Reference:
        1. <https://stackoverflow.com/a/2020083/18410348>
    """
    if not isinstance(obj, type):
        return fullname(type(obj))
    clazz: type = obj
    module: str = clazz.__module__
    if module == "builtins":
        return clazz.__qualname__
    return clazz.__module__ + "." + clazz.__qualname__


def is_instance_named(obj: Any, name: str | Sequence[str]) -> bool:
    if isinstance(name, str):
        name = [name]
    return any(fullname(clazz) in name for clazz in type(obj).mro())
