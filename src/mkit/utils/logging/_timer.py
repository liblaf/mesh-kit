import contextlib
import time
from collections.abc import Callable, Generator
from typing import Any

from loguru import logger


@contextlib.contextmanager
def timer(name: str | None = "") -> Generator[Callable[[], float], Any, None]:
    start: float = time.perf_counter()
    yield lambda: time.perf_counter() - start
    end: float = time.perf_counter()
    logger.opt(depth=1).debug("{} executed in {} sec.", name or "Block", end - start)
