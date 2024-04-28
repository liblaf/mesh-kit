from collections.abc import Callable

import typer

from mkit import log


def run(main: Callable) -> None:
    log.init()
    typer.run(main)
