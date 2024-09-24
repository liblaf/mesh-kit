from collections.abc import Callable
from pathlib import Path
from typing import Any

from mkit.typing import StrPath
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


def load(fpath: StrPath, ext: str | None = None, **kwargs) -> Any:
    fpath: Path = Path(fpath)
    if ext is None:
        ext = fpath.suffix
    if ext not in _READERS:
        msg: str = f"Unsupported file extension: {ext}"
        raise ValueError(msg)
    reader = _READERS[ext]
    return reader(fpath, **kwargs)


def save(data: Any, fpath: StrPath, ext: str | None = None, **kwargs) -> None:
    fpath: Path = Path(fpath)
    if ext is None:
        ext = fpath.suffix
    if ext not in _WRITERS:
        msg: str = f"Unsupported file extension: {ext}"
        raise ValueError(msg)
    writer = _WRITERS[ext]
    writer(data, fpath, **kwargs)
