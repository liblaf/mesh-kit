import pathlib
from typing import Annotated

import meshio
import numpy as np
import trimesh
import typer
from mkit import cli
from mkit import io as _io
from mkit.array import index
from numpy import typing as npt


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    mandible_file: Annotated[
        pathlib.Path, typer.Option("--mandible", dir_okay=False, writable=True)
    ],
    maxilla_file: Annotated[
        pathlib.Path, typer.Option("--maxilla", dir_okay=False, writable=True)
    ],
) -> None:
    skull_io: meshio.Mesh = meshio.read(input_file)
    skull_tr: trimesh.Trimesh = _io.to_trimesh(skull_io)
    mandible: trimesh.Trimesh
    maxilla: trimesh.Trimesh
    mandible, maxilla = skull_tr.split()
    process(skull_io, mandible, mandible_file)
    process(skull_io, maxilla, maxilla_file)


def process(
    skull_io: meshio.Mesh, component_tr: trimesh.Trimesh, output_file: pathlib.Path
) -> None:
    part_idx: npt.NDArray[np.intp] = index.position2index(
        skull_io.points, component_tr.vertices
    )
    part_io: meshio.Mesh = meshio.Mesh(
        points=component_tr.vertices,
        cells=[("triangle", component_tr.faces)],
        point_data={
            key: value[part_idx]  # pyright: ignore [reportIndexIssue]
            for key, value in skull_io.point_data.items()
        },
    )
    part_io.write(output_file)


if __name__ == "__main__":
    cli.run(main)
