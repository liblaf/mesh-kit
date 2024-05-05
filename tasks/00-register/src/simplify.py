import pathlib
from typing import Annotated

import trimesh
import typer
from mkit import cli


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True)],
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
    face_count: Annotated[int, typer.Option("--face-count")] = 10000,
) -> None:
    mesh: trimesh.Trimesh = trimesh.load(input_file)
    if face_count > 0 and mesh.faces.shape[0] > face_count:
        mesh = mesh.simplify_quadric_decimation(face_count)
    mesh.export(output_file)


if __name__ == "__main__":
    cli.run(main)
