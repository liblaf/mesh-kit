import pathlib
from typing import Annotated

import trimesh
import typer

from mesh_kit.common import cli


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    output_file: Annotated[pathlib.Path, typer.Argument(dir_okay=False, writable=True)],
) -> None:
    mesh: trimesh.Trimesh = trimesh.load(input_file)
    mesh = mesh.split()[0]
    mesh.export(output_file)


if __name__ == "__main__":
    cli.run(main)
