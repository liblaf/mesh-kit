import logging

from rich import logging as rich_logging

__all__ = ["init"]


def init(level: int = logging.DEBUG) -> None:
    handler: rich_logging.RichHandler = rich_logging.RichHandler(level=level)
    logging.basicConfig(
        format="%(message)s", datefmt="[%X]", level=level, handlers=[handler]
    )
