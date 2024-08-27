import functools
import pathlib
from collections.abc import Callable
from typing import ClassVar, ParamSpec, TypeVar

import confz
from icecream import ic
from loguru import logger

import mkit.logging


class BaseConfig(confz.BaseConfig):
    log_level: int | str = "INFO"
    log_file: pathlib.Path | None = None
    CONFIG_SOURCES: ClassVar = [confz.FileSource("params.yaml"), confz.CLArgSource()]

    def __post_init__(self) -> None:
        ic(self)


C = TypeVar("C", bound=BaseConfig)
P = ParamSpec("P")
T = TypeVar("T")


def cli(schema: type[C] = BaseConfig) -> Callable[[Callable[[C], T]], Callable[[], T]]:
    def wrapper(fn: Callable[[C], T]) -> Callable[[], T]:
        @functools.wraps(fn)
        def wrapped() -> T:
            cfg: C = schema()
            mkit.logging.init(cfg.log_level, cfg.log_file)
            logger.info("{}", cfg)
            return fn(cfg)

        return wrapped

    return wrapper


__all__ = ["BaseConfig", "cli"]
