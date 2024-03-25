from collections.abc import Callable

import typer

from mkit import log as _log


def run(main: Callable[..., None]) -> None:
    _log.init()
    typer.run(main)
