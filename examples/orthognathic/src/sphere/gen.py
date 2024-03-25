import pathlib
from typing import Annotated

import numpy as np
import trimesh
import typer
from mkit import cli as _cli
from numpy import typing as npt
from trimesh import bounds, creation


def main(
    output_file: Annotated[pathlib.Path, typer.Argument(dir_okay=False, writable=True)],
    radius: Annotated[float, typer.Option(min=0)] = 1,
    displacement: Annotated[float, typer.Option()] = 0,
) -> None:
    mesh: trimesh.Trimesh = creation.icosphere(radius=radius)
    if displacement:
        verts_mask: npt.NDArray = bounds.contains(
            [[-np.inf, -np.inf, -np.inf], [np.inf, 0, np.inf]], mesh.vertices
        )
        mesh.vertices[verts_mask] += [displacement, 0, 0]
    mesh.export(output_file)


if __name__ == "__main__":
    _cli.run(main)
