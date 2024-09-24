from typing import Any, TypeVar

import pydantic

from mkit.typing import StrPath
from mkit.utils import serialize

_C = TypeVar("_C", bound=pydantic.BaseModel)


def load_pydantic(
    cls: type[_C], fpath: StrPath, ext: str | None = None, **kwargs
) -> _C:
    data: Any = serialize.load(fpath, ext, **kwargs)
    return cls.model_validate(data)


def save_pydantic(
    data: pydantic.BaseModel, fpath: StrPath, ext: str | None = None, **kwargs
) -> None:
    serialize.save(data.model_dump(), fpath, ext, **kwargs)
