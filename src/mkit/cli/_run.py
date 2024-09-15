import inspect
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, TypeVar, Unpack

from loguru import logger

import mkit

_C = TypeVar("_C", bound=mkit.cli.BaseConfig)
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
    sig: inspect.Signature = inspect.signature(fn)
    annotation: type[_C] = sig.parameters["cfg"].annotation
    cfg: _C = annotation(**kwargs)
    mkit.logging.init(cfg.log_level, cfg.log_file)
    logger.info("{}", cfg)
    return fn(cfg)
