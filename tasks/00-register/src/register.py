import pathlib
from typing import Annotated, Optional

import meshio
import mkit.cli
import mkit.io
import numpy as np
import trimesh
import typer
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
    source_tr: trimesh.Trimesh = mkit.io.as_trimesh(source_io)
    target_tr: trimesh.Trimesh = mkit.io.as_trimesh(target_io)
    result: npt.NDArray[np.floating] = trimesh.registration.nricp_amberg(
        source_tr,
        target_tr,
        steps=[[0.02, 3, 0.5, 10], [0.007, 3, 0.5, 10], [0.002, 3, 0.5, 10]],
        distance_threshold=0.05,
    )
    source_io.points = result
    mkit.io.save(output_file, source_io)


if __name__ == "__main__":
    mkit.cli.run(main)
