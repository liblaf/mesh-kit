import logging
import sys
from pathlib import Path

import rich.traceback
from loguru import logger

from mkit.logging._handler import InterceptHandler
from mkit.typing import StrPath

DEFAULT_FILTER: dict[str | None, str | int | bool] = {
    "jax._src": logging.INFO,
    "numba.core": logging.INFO,
}


def init(level: str | int = logging.NOTSET, file: StrPath | None = None) -> None:
    rich.traceback.install(show_locals=True)
    logger.remove()
    logger.add(sys.stderr, level=level, filter=DEFAULT_FILTER)
    if file is not None:
        filepath: Path = Path(file)
        logger.add(filepath.open("w"), level=level, filter=DEFAULT_FILTER)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
