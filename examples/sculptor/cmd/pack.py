import pathlib
from typing import Annotated

import numpy as np
import trimesh
import typer
from numpy import typing as npt

from mesh_kit import cli as _cli
from mesh_kit import trimesh as _tri
from mesh_kit.io import trimesh as _io


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    output_file: Annotated[pathlib.Path, typer.Argument(dir_okay=False, writable=True)],
    *,
    mask_value: Annotated[float, typer.Option()] = 0.0,
) -> None:
    mesh: trimesh.Trimesh
    attrs: dict[str, npt.NDArray]
    mesh, attrs = _io.read(input_file, attr=True)
    if (
        landmark_file := input_file.with_stem(
            input_file.stem + "-landmarks"
        ).with_suffix(".txt")
    ).exists():
        positions: npt.NDArray = np.loadtxt(landmark_file)
        attrs["landmarks"] = _tri.pos2idx(mesh, positions)
    if (mask_file := input_file.with_stem(input_file.stem + "-mask")).exists():
        mask_mesh: trimesh.Trimesh = _io.read(mask_file)
        idx: npt.NDArray = _tri.pos2idx(mesh, mask_mesh.vertices)
        mask: npt.NDArray = np.ones(mesh.vertices.shape[0])
        mask[idx] = mask_value
        attrs["vert:distance-threshold"] = mask
    _io.write(output_file, mesh, attr=True, **attrs)


if __name__ == "__main__":
    _cli.run(main)
