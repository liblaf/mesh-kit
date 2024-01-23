import logging
import pathlib
from typing import Annotated

import numpy as np
import trimesh
import typer
from numpy import typing as npt
from trimesh import Trimesh

from mesh_kit.common import cli, path


def main(
    source_filepath: Annotated[
        pathlib.Path, typer.Argument(exists=True, dir_okay=False)
    ],
    target_filepath: Annotated[
        pathlib.Path, typer.Argument(exists=True, dir_okay=False)
    ],
    *,
    output_filepath: Annotated[
        pathlib.Path, typer.Option("--output", dir_okay=False, writable=True)
    ],
) -> None:
    source: trimesh.Trimesh = trimesh.load(source_filepath)
    # target: trimesh.Trimesh = trimesh.load(target_filepath)
    matrix: npt.NDArray
    transformed: npt.NDArray
    cost: float
    source_landmarks: npt.NDArray = np.loadtxt(path.landmarks(source_filepath))
    target_landmarks: npt.NDArray = np.loadtxt(path.landmarks(target_filepath))
    matrix, transformed, cost = trimesh.registration.procrustes(
        source_landmarks, target_landmarks, reflection=False
    )
    logging.info("Procrustes Cost: %f", cost)
    output: Trimesh = source.apply_transform(matrix)
    output.export(output_filepath)
    np.savetxt(path.landmarks(output_filepath), transformed)


if __name__ == "__main__":
    cli.run(main)
