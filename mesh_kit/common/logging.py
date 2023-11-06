import logging

from rich.logging import RichHandler


def init(level: int = logging.DEBUG) -> None:
    handler: RichHandler = RichHandler(level=level)
    logging.basicConfig(
        format="%(message)s", datefmt="[%X]", level=level, handlers=[handler]
    )
