from collections.abc import Callable

import typer

from mesh_kit import log as _log
from mesh_kit.typing import typechecked as _typechecked


@_typechecked
def run(main: typer.Typer | Callable) -> None:
    _log.init()
    if isinstance(main, typer.Typer):
        main()
    else:
        typer.run(main)
