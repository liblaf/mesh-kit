import pathlib
from typing import Annotated, Optional

import meshio
import mkit.ops.register
import numpy as np
import trimesh
import typer
from mkit import cli
from mkit import io as _io
from numpy import typing as npt


def main(
    source_file: Annotated[pathlib.Path, typer.Argument(exists=True)],
    target_file: Annotated[pathlib.Path, typer.Argument(exists=True)],
    *,
    output_file: Annotated[pathlib.Path, typer.Option("-o", "--output", writable=True)],
    record_dir: Annotated[
        Optional[pathlib.Path], typer.Option(file_okay=False, writable=True)
    ] = None,
) -> None:
    source_io: meshio.Mesh = meshio.read(source_file)
    target_io: meshio.Mesh = meshio.read(target_file)
    source_tr: trimesh.Trimesh = _io.as_trimesh(source_io)
    target_tr: trimesh.Trimesh = _io.as_trimesh(target_io)
    source_vert_mask: npt.NDArray[np.bool_] = np.asarray(
        source_io.point_data["mask"], np.bool_
    )
    mkit.ops.register.register(
        source_tr,
        target_tr,
        source_vert_mask=source_vert_mask,
        target_vert_mask=None,
        record_dir=record_dir,
    )


if __name__ == "__main__":
    cli.run(main)
