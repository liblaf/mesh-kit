import json
import pathlib
from typing import TYPE_CHECKING, Annotated, Optional

import trimesh
import typer

from mesh_kit.common import cli as _cli
from mesh_kit.io import landmark as _io_landmark
from mesh_kit.registration import config as _config
from mesh_kit.registration import landmark as _landmark
from mesh_kit.registration import nricp as _nricp

if TYPE_CHECKING:
    from numpy import typing as npt


def main(
    source_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    target_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
    config_file: Annotated[
        Optional[pathlib.Path],
        typer.Option("-c", "--config", exists=True, dir_okay=False),
    ] = None,
    record_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(exists=True, file_okay=False, writable=True),
    ] = None,
) -> None:
    source_mesh: trimesh.Trimesh = trimesh.load(source_file)
    source_positions: npt.NDArray = _io_landmark.read(source_file)
    target_mesh: trimesh.Trimesh = trimesh.load(target_file)
    target_positions: npt.NDArray = _io_landmark.read(target_file)
    source_landmarks: npt.NDArray = _landmark.position2index(
        mesh=source_mesh, positions=source_positions
    )
    if config_file:
        config = _config.Config(**json.loads(config_file.read_text()))
    else:
        config = None
    result_mesh: trimesh.Trimesh = _nricp.nricp_amberg(
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        source_landmarks=source_landmarks,
        target_positions=target_positions,
        config=config,
        record_dir=record_dir,
    )
    result_mesh.export(output_file)


if __name__ == "__main__":
    _cli.run(main)
