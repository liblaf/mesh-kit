import datetime
import pathlib
import shlex
from collections.abc import Callable, Sequence
from os import PathLike

import typer
from loguru import logger

import mkit.log
from mkit._typing import StrPath


def run(main: Callable) -> None:
    mkit.log.init()
    typer.run(main)


def up_to_date(
    outputs: StrPath | Sequence[StrPath | None] | None,
    inputs: StrPath | Sequence[StrPath | None] | None = None,
) -> None:
    return
    outputs = _as_list(outputs)
    inputs = _as_list(inputs)
    earlist_output: datetime.datetime = datetime.datetime.max
    for o in outputs:
        if not o.exists():
            return
        earlist_output = min(
            earlist_output, datetime.datetime.fromtimestamp(o.stat().st_mtime)
        )
    latest_input: datetime.datetime = datetime.datetime.min
    for i in inputs:
        if not i.exists():
            raise FileNotFoundError(i)
        latest_input = max(
            latest_input, datetime.datetime.fromtimestamp(i.stat().st_mtime)
        )
    if latest_input < earlist_output:
        logger.info("{} is up to date.", shlex.join([str(o) for o in outputs]))
    else:
        raise typer.Exit()


def _as_list(x: StrPath | Sequence[StrPath | None] | None) -> list[pathlib.Path]:
    if x is None:
        return []
    elif isinstance(x, str | PathLike):
        return [pathlib.Path(x)]
    else:
        return [pathlib.Path(i) for i in x if i is not None]
