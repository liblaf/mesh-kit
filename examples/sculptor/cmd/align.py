import pathlib
from typing import Annotated

import trimesh
import typer
from loguru import logger
from numpy import typing as npt
from trimesh import registration

from mesh_kit import cli as _cli
from mesh_kit.io import trimesh as _io
from mesh_kit.typing import check_shape as _check_shape


def main(
    source_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    target_file: Annotated[pathlib.Path, typer.Argument(dir_okay=False, writable=True)],
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
) -> None:
    source_mesh: trimesh.Trimesh
    source_attrs: dict[str, npt.NDArray]
    source_mesh, source_attrs = _io.read(source_file, attr=True)
    target_mesh: trimesh.Trimesh
    target_attrs: dict[str, npt.NDArray]
    target_mesh, target_attrs = _io.read(target_file, attr=True)
    num_landmarks: int = source_attrs["landmarks"].shape[0]
    source_landmarks: npt.NDArray = _check_shape(
        source_attrs["landmarks"], (num_landmarks,)
    )
    target_landmarks: npt.NDArray = _check_shape(
        target_attrs["landmarks"], (num_landmarks,)
    )
    matrix: npt.NDArray
    transformed: npt.NDArray
    cost: float
    matrix, transformed, cost = registration.procrustes(
        source_mesh.vertices[source_landmarks], target_mesh.vertices[target_landmarks]
    )
    matrix = _check_shape(matrix, (4, 4))
    logger.info("procrustes(): cost = {}", cost)
    output_mesh: trimesh.Trimesh = source_mesh.apply_transform(matrix)
    output_attrs: dict[str, npt.NDArray] = source_attrs.copy()
    _io.write(output_file, output_mesh, attr=True, **output_attrs)


if __name__ == "__main__":
    _cli.run(main)
