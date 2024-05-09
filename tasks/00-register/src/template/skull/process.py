import pathlib
from typing import Annotated

import mkit.cli
import mkit.io
import numpy as np
import scipy
import scipy.interpolate
import trimesh
import typer
from numpy import typing as npt


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
    mandible_file: Annotated[
        pathlib.Path, typer.Option("--mandible", dir_okay=False, writable=True)
    ],
    maxilla_file: Annotated[
        pathlib.Path, typer.Option("--maxilla", dir_okay=False, writable=True)
    ],
    threshold: Annotated[float, typer.Option()] = 0.02,
) -> None:
    mkit.cli.up_to_date(
        [output_file, mandible_file, maxilla_file], [__file__, input_file]
    )
    skull: trimesh.Trimesh = trimesh.load(input_file)
    mandible: trimesh.Trimesh
    maxilla: trimesh.Trimesh
    mandible, maxilla = split_skull(skull)
    closest: npt.NDArray[np.float64]
    distance: npt.NDArray[np.float64]
    triangle_id: npt.NDArray[np.intp]
    closest, distance, triangle_id = maxilla.nearest.on_surface(mandible.vertices)
    mandible_mask: npt.NDArray[np.bool_] = distance > threshold * skull.scale
    closest, distance, triangle_id = mandible.nearest.on_surface(maxilla.vertices)
    maxilla_mask: npt.NDArray[np.bool_] = distance > threshold * skull.scale
    skull_mask: npt.NDArray[np.bool_] = scipy.interpolate.griddata(
        np.vstack([mandible.vertices, maxilla.vertices]),
        np.concatenate([mandible_mask, maxilla_mask]),
        skull.vertices,
        method="nearest",
    ).astype(np.bool_)
    mkit.io.save(output_file, skull, point_data={"mask": skull_mask.astype(np.int8)})
    mkit.io.save(
        mandible_file, mandible, point_data={"mask": mandible_mask.astype(np.int8)}
    )
    mkit.io.save(
        maxilla_file, maxilla, point_data={"mask": maxilla_mask.astype(np.int8)}
    )


def split_skull(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    mandible: trimesh.Trimesh
    maxilla: trimesh.Trimesh
    mandible, maxilla = mesh.split()
    if mandible.area > maxilla.area:
        mandible, maxilla = maxilla, mandible
    return mandible, maxilla


if __name__ == "__main__":
    mkit.cli.run(main)
