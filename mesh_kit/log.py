import functools
import inspect
import logging
import os
import sys
import time
from typing import Callable, ParamSpec, TypeVar

from loguru import logger

from mesh_kit.typing import typechecked as _typechecked


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


@_typechecked
def init(level: int | str = logging.NOTSET) -> None:
    if level == logging.NOTSET:
        level = os.getenv("LOG_LEVEL", logging.NOTSET)
    if level == logging.NOTSET:
        level = logging.DEBUG if __debug__ else logging.INFO
    logging.basicConfig(handlers=[InterceptHandler()], level=level, force=True)
    logger.remove()
    logger.add(sys.stderr, level=level)


class Timer:
    name: str | None
    depth: int
    start_time: float
    end_time: float

    def __init__(self, name: str | None = None, depth: int = 1) -> None:
        self.name = name
        self.depth = depth

    def __enter__(self) -> None:
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        if self.name:
            logger.opt(depth=self.depth).debug(
                "{}: {}s", self.name, self.end_time - self.start_time
            )
        else:
            logger.opt(depth=self.depth).debug("{}s", self.end_time - self.start_time)


P = ParamSpec("P")
T = TypeVar("T")


def timeit(func: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__name__ + "()", depth=2):
            return func(*args, **kwargs)

    return wrapper  # pyright: ignore
