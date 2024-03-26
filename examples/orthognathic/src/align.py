import pathlib
from typing import Annotated, Any

import numpy as np
import trimesh
import typer
from loguru import logger
from mkit import cli as _cli
from mkit.typing import as_any as _a
from numpy import typing as npt
from trimesh import registration


def main(
    source_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    target_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False)
    ],
) -> None:
    _: Any
    source_mesh: trimesh.Trimesh = _a(trimesh.load(source_file))
    target_mesh: trimesh.Trimesh = _a(trimesh.load(target_file))
    matrix: npt.NDArray
    cost: float
    matrix, _, cost = registration.procrustes(
        source_mesh.vertices, target_mesh.vertices, reflection=False, scale=False
    )
    logger.info("procrustes(): cost = {}", cost)
    np.save(output_file, matrix)


if __name__ == "__main__":
    _cli.run(main)
