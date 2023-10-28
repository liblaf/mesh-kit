from pathlib import Path
from typing import Annotated, cast

import trimesh
from trimesh import Trimesh
from typer import Argument

from mesh_kit.common.cli import run


def main(
    input_path: Annotated[Path, Argument(exists=True, dir_okay=False)],
    output_path: Annotated[Path, Argument(dir_okay=False, writable=True)],
) -> None:
    mesh: Trimesh = cast(Trimesh, trimesh.load(input_path))
    mesh.export(output_path)


if __name__ == "__main__":
    run(main)
