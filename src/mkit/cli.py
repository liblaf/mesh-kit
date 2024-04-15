from collections.abc import Callable

import typer

from mkit import _log


def run(main: Callable) -> None:
    _log.init()
    typer.run(main)
