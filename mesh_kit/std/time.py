import functools
import time
from collections.abc import Callable
from typing import Any, Optional

from loguru import logger


class PerfCounter:
    depth: int = 1
    name: Optional[str] = None
    start: float
    stop: float

    def __init__(self, name: Optional[str] = None, depth: int = 1) -> None:
        self.name = name
        self.depth = depth

    def __enter__(self) -> None:
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop = time.perf_counter()
        if self.name is None:
            logger.opt(depth=self.depth).debug("Time: {}s", self.stop - self.start)
        else:
            logger.opt(depth=self.depth).debug(
                "{} Time: {}s", self.name, self.stop - self.start
            )


def timeit(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        with PerfCounter(name=func.__name__ + "()", depth=2):
            return func(*args, **kwargs)

    return wrapper
