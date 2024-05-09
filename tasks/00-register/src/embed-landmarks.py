import pathlib
from typing import Annotated

import meshio
import mkit
import mkit.array
import mkit.array.points
import mkit.cli
import mkit.io
import numpy as np
import typer
from git import Optional
from numpy import typing as npt


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    landmarks_file: Annotated[
        Optional[pathlib.Path], typer.Option("-l", "--landmarks")
    ] = None,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
) -> None:
    mesh: meshio.Mesh = mkit.io.load_meshio(input_file)
    landmarks: npt.NDArray[np.integer] = load_landmarks(mesh, landmarks_file)
    mkit.io.save(output_file, mesh, field_data={"landmarks": landmarks})


def load_landmarks(
    mesh: meshio.Mesh, landmarks_file: pathlib.Path | None
) -> npt.NDArray[np.integer]:
    if landmarks_file is None or not landmarks_file.exists():
        return np.empty((0, 3), dtype=np.intp)
    pos: npt.NDArray[np.floating] = np.loadtxt(landmarks_file)
    idx: npt.NDArray[np.integer] = mkit.array.points.position_to_index(mesh.points, pos)
    return idx


if __name__ == "__main__":
    mkit.cli.run(main)
