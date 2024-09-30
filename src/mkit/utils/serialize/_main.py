import pathlib
from collections.abc import Callable
from typing import Any

import pydantic

from mkit.utils import serialize

_READERS: dict[str, Callable[..., Any]] = {
    ".json": serialize.load_json,
    ".toml": serialize.load_toml,
    ".yaml": serialize.load_yaml,
}


_WRITERS: dict[str, Callable[..., None]] = {
    ".json": serialize.save_json,
    ".toml": serialize.save_toml,
    ".yaml": serialize.save_yaml,
}


@pydantic.validate_call
def load(fpath: pydantic.FilePath, ext: str | None = None, **kwargs) -> Any:
    if ext is None:
        ext = fpath.suffix
    if ext not in _READERS:
        msg: str = f"Unsupported file extension: {ext}"
        raise ValueError(msg)
    reader = _READERS[ext]
    return reader(fpath, **kwargs)


@pydantic.validate_call
def save(data: Any, fpath: pathlib.Path, ext: str | None = None, **kwargs) -> None:
    if ext is None:
        ext = fpath.suffix
    if ext not in _WRITERS:
        msg: str = f"Unsupported file extension: {ext}"
        raise ValueError(msg)
    writer = _WRITERS[ext]
    writer(data, fpath, **kwargs)
