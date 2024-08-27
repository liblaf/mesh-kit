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


def run(fn: Callable[[C], T], schema: type[C] = BaseConfig, **kwargs) -> T:
    cfg: C = schema(**kwargs)
    mkit.logging.init(cfg.log_level, cfg.log_file)
    logger.info("{}", cfg)
    return fn(cfg)


__all__ = ["BaseConfig", "run"]
