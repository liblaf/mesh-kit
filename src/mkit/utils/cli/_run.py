from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, TypeVar, Unpack, get_type_hints

import rich.traceback
from loguru import logger

import mkit.utils as mu

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_C = TypeVar("_C", bound=mu.cli.BaseConfig)
_T = TypeVar("_T")


class Kwargs(TypedDict, total=False):
    log_file: Path
    log_level: int | str


def auto_run(
    **kwargs: Unpack[Kwargs],
) -> Callable[[Callable[[_C], _T]], Callable[[_C], _T]]:
    def wrapper(fn: Callable[[_C], _T]) -> Callable[[_C], _T]:
        if fn.__module__ == "__main__":
            run(fn, **kwargs)
        return fn

    return wrapper


def run(fn: Callable[[_C], _T], **kwargs: Unpack[Kwargs]) -> _T:
    rich.traceback.install(show_locals=True)
    cls: type[_C] = get_type_hints(fn)["cfg"]
    cfg: _C = cls(**kwargs)
    mu.logging.init(cfg.log_level, cfg.log_file)
    logger.info("{}", cfg)
    return fn(cfg)
