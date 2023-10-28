from collections.abc import Callable
from typing import Optional

import typer
from typer import Typer


def add_command(app: Typer, command: Callable, name: Optional[str] = None) -> None:
    if isinstance(command, Typer):
        app.add_typer(typer_instance=command, name=name)
    else:
        app.command(name=name)(command)


def run(command: Callable) -> None:
    if isinstance(command, Typer):
        command()
    else:
        typer.run(command)
