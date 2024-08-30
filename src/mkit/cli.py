import inspect
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


def run(
    fn: Callable[[C], T],
    log_level: int | str = "INFO",
    log_file: pathlib.Path | None = None,
    **kwargs,
) -> T:
    sig: inspect.Signature = inspect.signature(fn)
    annotation: type[C] = sig.parameters["cfg"].annotation
    kwargs.update({"log_level": log_level, "log_file": log_file})
    cfg: C = annotation(
        config_sources=[
            confz.FileSource("params.yaml", optional=True),
            confz.CLArgSource(),
            confz.DataSource(kwargs),
        ]
    )
    mkit.logging.init(cfg.log_level, cfg.log_file)
    logger.info("{}", cfg)
    return fn(cfg)


__all__ = ["BaseConfig", "run"]
