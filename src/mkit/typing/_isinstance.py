from collections.abc import Sequence
from typing import Any

import mkit.utils as mu


def full_name(obj: Any) -> str:
    """Returns the fully qualified name of the given object.

    Args:
        obj: The object whose fully qualified name is to be returned.

    Returns:
        The fully qualified name of the object.

    Reference:
        1. <https://stackoverflow.com/a/2020083/18410348>
    """
    if not isinstance(obj, type):
        return full_name(type(obj))
    clazz: type = obj
    module: str = clazz.__module__
    if module == "builtins":
        return clazz.__qualname__
    return clazz.__module__ + "." + clazz.__qualname__


def is_class_named(cls: type, name: str | Sequence[str]) -> bool:
    return any((full_name(cls) in mu.flatten(name)) for cls in cls.__mro__)


def is_class_named_partial(cls: type, name: str | Sequence[str]) -> bool:
    for clazz in cls.__mro__:
        class_parts: list[str] = full_name(clazz).split(".")
        for n in mu.flatten(name):
            name_parts: list[str] = n.split(".")
            if mu.is_subsequence(name_parts, class_parts):
                return True
    return False


def is_instance_named(obj: Any, name: str | Sequence[str]) -> bool:
    return is_class_named(type(obj), name)


def is_instance_named_partial(obj: Any, name: str | Sequence[str]) -> bool:
    return is_class_named_partial(type(obj), name)


def is_named(obj: Any, name: str | Sequence[str]) -> bool:
    if isinstance(obj, type):
        return is_class_named(obj, name)
    return is_instance_named(obj, name)


def is_named_partial(obj: Any, name: str | Sequence[str]) -> bool:
    if isinstance(obj, type):
        return is_class_named_partial(obj, name)
    return is_instance_named_partial(obj, name)
