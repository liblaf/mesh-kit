from collections.abc import Callable

import typer

from mesh_kit import log as _log


def run(main: Callable) -> None:
    _log.init()
    typer.run(main)
