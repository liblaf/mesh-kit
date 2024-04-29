import pathlib
from typing import Annotated

import trimesh
import typer
from mkit import cli


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True)],
    output_file: Annotated[pathlib.Path, typer.Argument(writable=True)],
    *,
    face_count: Annotated[int, typer.Option("--face-count")] = 10000,
) -> None:
    mesh: trimesh.Trimesh = trimesh.load(input_file)
    if face_count > 0:
        mesh = simplify(mesh, face_count)
    mesh.export(output_file)


def simplify(mesh: trimesh.Trimesh, face_count: int) -> trimesh.Trimesh:
    if mesh.faces.shape[0] <= face_count:
        return mesh
    return mesh.simplify_quadric_decimation(face_count)


if __name__ == "__main__":
    cli.run(main)
