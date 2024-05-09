import pathlib
from typing import Annotated

import igl
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
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
) -> None:
    tetra: meshio.Mesh = mkit.io.load_meshio(input_file)
    faces: npt.NDArray[np.integer] = igl.boundary_facets(tetra.get_cells_type("tetra"))  # pyright: ignore [reportAttributeAccessIssue]
    triangle = trimesh.Trimesh(tetra.points, faces)
    mkit.io.save(output_file, triangle)


if __name__ == "__main__":
    mkit.cli.run(main)
