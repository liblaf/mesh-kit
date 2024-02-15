import functools
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import numpy as np
from loguru import logger
from scipy import sparse

from mesh_kit.std import time as _time

P = ParamSpec("P")
T = TypeVar("T")


def cache(func: Callable[P, T]) -> Callable[P, T]:
    cache: dict[int, Any] = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _time.PerfCounter():
            key: int = 0
            for arg in args:
                key ^= _hash(arg)
            for k, v in kwargs.items():
                key ^= _hash(k) ^ _hash(v)
        if key not in cache:
            logger.debug("Cache Miss")
            cache[key] = func(*args, **kwargs)
        else:
            logger.debug("Cache Hit")
        return cache[key]

    return wrapper


def _hash(obj: Any) -> int:
    try:
        return hash(obj)
    except TypeError as e:
        match obj:
            case np.ndarray():
                return hash(obj.dumps())
            case sparse.coo_matrix() | sparse.csr_matrix():
                return _hash(obj.data)
            case obj:
                logger.error(e)
                return id(obj)
