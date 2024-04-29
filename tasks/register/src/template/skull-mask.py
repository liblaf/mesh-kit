import pathlib
from typing import Annotated

import numpy as np
import trimesh
import typer
from mkit import cli
from mkit import io as _io
from numpy import typing as npt
from scipy import interpolate

THRESHOLD: float = 0.02


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    output_file: Annotated[pathlib.Path, typer.Argument(dir_okay=False, writable=True)],
) -> None:
    skull: trimesh.Trimesh = trimesh.load(input_file)
    mandible: trimesh.Trimesh
    maxilla: trimesh.Trimesh
    mandible, maxilla = skull.split()
    closest: npt.NDArray[np.float64]
    distance: npt.NDArray[np.float64]
    triangle_id: npt.NDArray[np.intp]
    closest, distance, triangle_id = maxilla.nearest.on_surface(mandible.vertices)
    mandible_mask: npt.NDArray[np.bool_] = distance > THRESHOLD * skull.scale
    closest, distance, triangle_id = mandible.nearest.on_surface(maxilla.vertices)
    maxilla_mask: npt.NDArray[np.bool_] = distance > THRESHOLD * skull.scale
    skull_mask: npt.NDArray[np.bool_] = interpolate.griddata(
        np.vstack((mandible.vertices, maxilla.vertices)),
        np.concatenate((mandible_mask, maxilla_mask)),
        skull.vertices,
        method="nearest",
    ).astype(np.bool_)
    _io.save(output_file, skull, point_data={"mask": skull_mask.astype(np.int8)})


if __name__ == "__main__":
    cli.run(main)
