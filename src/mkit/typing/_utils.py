from collections.abc import Sequence
from typing import Any


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


def full_name_parts(obj: Any) -> list[str]:
    return full_name(obj).split(".")


def is_instance_named(obj: Any, name: str | Sequence[str]) -> bool:
    if isinstance(name, str):
        name = [name]
    return any((full_name(clazz) in name) for clazz in type(obj).mro())


def is_instance_named_partial(obj: Any, name: str | Sequence[str]) -> bool:
    if isinstance(name, str):
        name = [name]
    for clazz in type(obj).mro():
        obj_parts: list[str] = full_name_parts(clazz)
        for n in name:
            class_parts: list[str] = n.split(".")
            if is_subsequence(class_parts, obj_parts):
                return True
    return False


def is_subsequence(a: Sequence[Any], b: Sequence[Any]) -> bool:
    i: int = 0
    j: int = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
        j += 1
    return i == len(a)
