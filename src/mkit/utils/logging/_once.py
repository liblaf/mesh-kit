import functools

from loguru import logger


@functools.cache
def warning_once(message: str, *args, **kwargs) -> None:
    logger.warning(message, *args, **kwargs)
