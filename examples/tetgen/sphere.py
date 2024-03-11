import pathlib
from typing import Annotated

import trimesh
import typer
from mesh_kit import cli
from trimesh import creation


def main(
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
    radius: Annotated[float, typer.Option()] = 1.0,
) -> None:
    sphere: trimesh.Trimesh = creation.icosphere(radius=radius)
    sphere.export(output_file)


if __name__ == "__main__":
    cli.run(main)
