import pathlib
from typing import Annotated

import meshio
import mkit.cli
import mkit.io
import numpy as np
import trimesh
import typer
from numpy import typing as npt


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    transform_file: Annotated[
        pathlib.Path, typer.Option("--transform", exists=True, dir_okay=False)
    ],
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
    inverse: Annotated[bool, typer.Option()] = False,
) -> None:
    mkit.cli.up_to_date(output_file, [__file__, input_file, transform_file])
    mesh: meshio.Mesh = mkit.io.load_meshio(input_file)
    transform: npt.NDArray[np.floating] = np.load(transform_file)
    if inverse:
        transform = trimesh.transformations.inverse_matrix(transform)
    mesh.points = trimesh.transform_points(mesh.points, transform)
    mkit.io.save(output_file, mesh)


if __name__ == "__main__":
    mkit.cli.run(main)
