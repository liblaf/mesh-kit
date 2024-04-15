from os import PathLike
from typing import Any, AnyStr, TypeAlias

StrPath: TypeAlias = str | PathLike[str]
BytesPath: TypeAlias = bytes | PathLike[bytes]
GenericPath: TypeAlias = AnyStr | PathLike[AnyStr]
StrOrBytesPath: TypeAlias = str | bytes | PathLike[str] | PathLike[bytes]


def as_any(val: Any) -> Any:
    return val
