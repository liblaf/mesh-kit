import pathlib
from typing import Annotated, Optional

import meshio
import numpy as np
import trimesh
import typer
from loguru import logger
from mkit import cli
from mkit import io as _io
from numpy import typing as npt
from trimesh import registration


def main(
    source_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    target_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    initial_transform_file: Annotated[
        Optional[pathlib.Path],
        typer.Option("--initial-transform", exists=True, dir_okay=False),
    ] = None,
    output_mesh_file: Annotated[
        Optional[pathlib.Path],
        typer.Option("--output-mesh", dir_okay=False, writable=True),
    ],
    output_transform_file: Annotated[
        Optional[pathlib.Path],
        typer.Option("--output-transform", dir_okay=False, writable=True),
    ] = None,
) -> None:
    source_io: meshio.Mesh = meshio.read(source_file)
    target_io: meshio.Mesh = meshio.read(target_file)
    initial_transform: npt.NDArray[np.float64] | None = (
        np.load(initial_transform_file) if initial_transform_file is not None else None
    )
    source_tr: trimesh.Trimesh = _io.to_trimesh(source_io)
    target_tr: trimesh.Trimesh = _io.to_trimesh(target_io)
    source2target: npt.NDArray[np.float64]
    cost: float
    source2target, cost = align(
        source_tr, target_tr, initial_transform=initial_transform
    )
    if output_mesh_file is not None:
        source_tr.apply_transform(source2target)
        source_io.points = source_tr.vertices
        source_io.write(output_mesh_file)
    if output_transform_file is not None:
        np.save(output_transform_file, source2target)


def align(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    initial_transform: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.float64], float]:
    if initial_transform is None:
        source2target: npt.NDArray[np.float64]
        cost: float
        source2target, cost = registration.mesh_other(
            source, target, scale=True, icp_first=50, icp_final=100
        )
    else:
        source2target: npt.NDArray[np.float64]
        transformed: npt.NDArray[np.float64]
        cost: float
        source2target, transformed, cost = registration.icp(
            source.sample(10000),
            target.sample(10000),
            initial=initial_transform,
            max_iterations=100,
        )
    logger.info("Align Cost: {}", cost)
    return source2target, cost


if __name__ == "__main__":
    cli.run(main)
