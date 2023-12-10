import logging
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import trimesh
from numpy.typing import NDArray
from trimesh import Trimesh
from typer import Argument, Option

from mesh_kit.common.cli import run
from mesh_kit.common.path import landmarks_filepath


def main(
    source_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    target_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    *,
    output_filepath: Annotated[Path, Option("--output", dir_okay=False, writable=True)],
) -> None:
    source: Trimesh = cast(Trimesh, trimesh.load(source_filepath))
    # target: Trimesh = cast(Trimesh, trimesh.load(target_filepath))
    matrix: NDArray
    transformed: NDArray
    cost: float
    source_landmarks: NDArray = np.loadtxt(landmarks_filepath(source_filepath))
    target_landmarks: NDArray = np.loadtxt(landmarks_filepath(target_filepath))
    matrix, transformed, cost = trimesh.registration.procrustes(
        source_landmarks, target_landmarks, reflection=False
    )
    logging.info(f"Procrustes cost: {cost}")
    output: Trimesh = source.apply_transform(matrix)
    output.export(output_filepath)
    np.savetxt(landmarks_filepath(output_filepath), transformed)


if __name__ == "__main__":
    run(main)
