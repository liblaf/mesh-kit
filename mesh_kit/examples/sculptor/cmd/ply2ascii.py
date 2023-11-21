from pathlib import Path
from typing import Annotated, cast

import trimesh
import trimesh.exchange
from trimesh import Trimesh
from typer import Argument

from mesh_kit.common.cli import run


def main(
    input_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    output_filepath: Annotated[Path, Argument(dir_okay=False, writable=True)],
) -> None:
    mesh: Trimesh = cast(Trimesh, trimesh.load(input_filepath))
    mesh.export(output_filepath, encoding="ascii")


if __name__ == "__main__":
    run(main)
