import pathlib
from typing import Annotated

import trimesh
import typer
from mesh_kit import cli
from trimesh import creation


def main(
    output_dir: Annotated[
        pathlib.Path, typer.Argument(exists=True, file_okay=False, writable=True)
    ],
) -> None:
    face: trimesh.Trimesh = creation.icosphere(radius=2)
    face.export(output_dir / "pre-face.ply")
    skull: trimesh.Trimesh = creation.icosphere(radius=1)
    skull.export(output_dir / "pre-skull.ply")
    skull.vertices[skull.vertices[:, 1] < 0] += [0.1, 0, 0]
    skull.export(output_dir / "post-skull.ply")


if __name__ == "__main__":
    cli.run(main)
