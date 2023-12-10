from collections.abc import Callable
from typing import Optional

import typer

from mesh_kit.common import logging as mk_logging

__all__ = ["add_command", "run"]


def add_command(
    app: typer.Typer, command: Callable, name: Optional[str] = None
) -> None:
    if isinstance(command, typer.Typer):
        app.add_typer(typer_instance=command, name=name)
    else:
        app.command(name=name)(command)


def run(command: Callable) -> None:
    mk_logging.init()
    if isinstance(command, typer.Typer):
        command()
    else:
        typer.run(command)
