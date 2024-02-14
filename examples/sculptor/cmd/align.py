import pathlib
from typing import Annotated

import trimesh
import typer
from loguru import logger
from numpy import typing as npt
from trimesh import registration

from mesh_kit.common import cli as _cli
from mesh_kit.common import testing
from mesh_kit.io import landmark as _landmark


def main(
    source_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    target_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
) -> None:
    source_mesh: trimesh.Trimesh = trimesh.load(source_file)
    # target_mesh: trimesh.Trimesh = trimesh.load(target_file)
    source_landmarks: npt.NDArray = _landmark.read(source_file)
    testing.assert_shape(source_landmarks.shape, (-1, 3))
    target_landmarks: npt.NDArray = _landmark.read(target_file)
    testing.assert_shape(target_landmarks.shape, source_landmarks.shape)
    matrix: npt.NDArray
    transformed: npt.NDArray
    cost: float
    matrix, transformed, cost = registration.procrustes(
        source_landmarks, target_landmarks
    )
    testing.assert_shape(matrix.shape, (4, 4))
    testing.assert_shape(transformed.shape, source_landmarks.shape)
    logger.info("procrustes() cost: {}", cost)
    aligned_mesh: trimesh.Trimesh = source_mesh.apply_transform(matrix)
    aligned_mesh.export(output_file)
    _landmark.write(output_file, transformed)


if __name__ == "__main__":
    _cli.run(main)
