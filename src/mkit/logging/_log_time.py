import functools
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from loguru import logger

_P = ParamSpec("_P")
_T = TypeVar("_T")


def log_time(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    @functools.wraps(fn)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        start: float = time.perf_counter()
        result: _T = fn(*args, **kwargs)
        end: float = time.perf_counter()
        logger.opt(depth=1).debug("{}() executed in {} sec.", fn.__name__, end - start)
        return result

    return wrapped
