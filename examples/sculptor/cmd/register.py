import pathlib
from typing import Annotated, Optional

import trimesh
import typer
from numpy import typing as npt

from mesh_kit.common import cli
from mesh_kit.io import landmarks as io_landmarks
from mesh_kit.registration import landmarks as registration_landmarks
from mesh_kit.registration import nricp


def main(
    source_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    target_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    output_file: Annotated[pathlib.Path, typer.Option("--output", dir_okay=False)],
    record_dir: Annotated[
        Optional[pathlib.Path], typer.Option("--record", exists=True, file_okay=False)
    ] = None,
) -> None:
    source_mesh: trimesh.Trimesh = trimesh.load(source_file)
    target_mesh: trimesh.Trimesh = trimesh.load(target_file)
    source_positions: npt.NDArray = io_landmarks.read(source_file)
    target_positions: npt.NDArray = io_landmarks.read(target_file)
    source_landmarks: npt.NDArray = registration_landmarks.position_to_index(
        mesh=source_mesh, position=source_positions, workers=8
    )
    result: trimesh.Trimesh = nricp.nricp_amber(
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        source_landmarks=source_landmarks,
        target_positions=target_positions,
        record_dir=record_dir,
    )
    result.export(output_file, encoding="ascii")


if __name__ == "__main__":
    cli.run(main)
