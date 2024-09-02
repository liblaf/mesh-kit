import logging
import sys
from pathlib import Path

import rich.traceback
from loguru import logger

from mkit.logging._handler import InterceptHandler
from mkit.typing import StrPath

FILTER: dict[str | None, str | int | bool] = {
    "jax._src": logging.INFO,
    "numba.core": logging.INFO,
}


def init(level: str | int = logging.NOTSET, file: StrPath | None = None) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, filter=FILTER)
    if file is not None:
        filepath: Path = Path(file)
        logger.add(filepath.open("w"), level=level, filter=FILTER)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    rich.traceback.install(show_locals=True)
