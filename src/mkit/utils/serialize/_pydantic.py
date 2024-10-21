from typing import Any, TypeVar

import pydantic

from mkit.typing import StrPath
from mkit.utils import serialize

_C = TypeVar("_C", bound=pydantic.BaseModel)


def load_pydantic(fpath: StrPath, cls: type[_C], *, ext: str | None = None) -> _C:
    data: Any = serialize.load(fpath, ext=ext)
    return cls.model_validate(data)


def save_pydantic(
    fpath: StrPath, data: pydantic.BaseModel, *, ext: str | None = None
) -> None:
    serialize.save(fpath, data.model_dump(), ext=ext)
