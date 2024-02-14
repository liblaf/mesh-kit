from collections.abc import Callable
from typing import Any

import typer

from mesh_kit.common import log


def run(main: Callable[..., Any]) -> None:
    log.init()
    typer.run(main)
