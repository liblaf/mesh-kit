import json
import pathlib
from typing import Annotated, Optional

import trimesh
import typer
from numpy import typing as npt

from mesh_kit import cli as _cli
from mesh_kit.io import trimesh as _io
from mesh_kit.register import config as _config
from mesh_kit.register import nricp as _nricp


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
    source_mesh: trimesh.Trimesh
    source_attrs: dict[str, npt.NDArray]
    source_mesh, source_attrs = _io.read(source_file, attr=True)
    target_mesh: trimesh.Trimesh
    target_attrs: dict[str, npt.NDArray]
    target_mesh, target_attrs = _io.read(target_file, attr=True)
    if config_file:
        config = _config.Config(**json.loads(config_file.read_text()))
    else:
        config = None
    result_mesh: trimesh.Trimesh = _nricp.nricp_amberg(
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        source_attrs=source_attrs,
        target_attrs=target_attrs,
        config=config,
        record_dir=record_dir,
    )
    _io.write(output_file, result_mesh)


if __name__ == "__main__":
    _cli.run(main)
