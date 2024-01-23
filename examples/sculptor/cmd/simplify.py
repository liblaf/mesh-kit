from pathlib import Path
from typing import Annotated

import trimesh
import typer

from mesh_kit.common import cli


def main(
    input_filepath: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_filepath: Annotated[Path, typer.Argument(dir_okay=False, writable=True)],
    *,
    face_count: Annotated[int, typer.Option()] = -1,
) -> None:
    source: trimesh.Trimesh = trimesh.load(input_filepath)
    output: trimesh.Trimesh = (
        source.simplify_quadric_decimation(face_count=face_count)
        if face_count > 0
        else source
    )
    output.export(output_filepath)


if __name__ == "__main__":
    cli.run(main)
