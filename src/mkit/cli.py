from collections.abc import Callable

import typer

import mkit.log


def run(main: Callable) -> None:
    mkit.log.init()
    typer.run(main)
