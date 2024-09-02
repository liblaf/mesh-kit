import inspect
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import confz
from loguru import logger

import mkit

_C = TypeVar("_C", bound=mkit.cli.BaseConfig)
_T = TypeVar("_T")


def run(
    fn: Callable[[_C], _T],
    log_level: int | str = "INFO",
    log_file: Path | None = None,
    **kwargs,
) -> _T:
    sig: inspect.Signature = inspect.signature(fn)
    annotation: type[_C] = sig.parameters["cfg"].annotation
    kwargs.update({"log_level": log_level, "log_file": log_file})
    cfg: _C = annotation(
        config_sources=[
            confz.FileSource("params.yaml", optional=True),
            confz.CLArgSource(),
            confz.DataSource(kwargs),
        ]
    )
    mkit.logging.init(cfg.log_level, cfg.log_file)
    logger.info("{}", cfg)
    return fn(cfg)
