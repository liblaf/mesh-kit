import pathlib
from typing import Annotated

import mkit.cli
import typer


def main(
    face_file: Annotated[pathlib.Path, typer.Option("--face", exists=True)],
    mandible_file: Annotated[pathlib.Path, typer.Option("--mandible", exists=True)],
    maxilla_file: Annotated[pathlib.Path, typer.Option("--maxilla", exists=True)],
) -> None:
    pass


if __name__ == "__main__":
    mkit.cli.run(main)
