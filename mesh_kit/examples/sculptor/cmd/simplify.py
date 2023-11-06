from pathlib import Path
from typing import Annotated, cast

import trimesh
from trimesh import Trimesh
from typer import Argument, Option

from mesh_kit.common.cli import run


def main(
    input_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    output_filepath: Annotated[Path, Argument(dir_okay=False, writable=True)],
    *,
    face_count: Annotated[int, Option()] = -1,
) -> None:
    source: Trimesh = cast(Trimesh, trimesh.load(input_filepath))
    output: Trimesh = (
        source.simplify_quadric_decimation(face_count=face_count)
        if face_count > 0
        else source
    )
    output.export(output_filepath)


if __name__ == "__main__":
    run(main)
