import pathlib
from typing import Annotated

import mkit.cli
import mkit.io
import trimesh
import typer


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True)],
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
    face_count: Annotated[int, typer.Option("--face-count")] = 10000,
) -> None:
    mkit.cli.up_to_date(output_file, [__file__, input_file])
    mesh: trimesh.Trimesh = mkit.io.load_trimesh(input_file)
    if face_count > 0 and mesh.faces.shape[0] > face_count:
        mesh = mesh.simplify_quadric_decimation(face_count)
    mkit.io.save(output_file, mesh)


if __name__ == "__main__":
    mkit.cli.run(main)
