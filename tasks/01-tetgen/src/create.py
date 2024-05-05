import pathlib
from typing import Annotated

import meshio
import mkit.cli
import mkit.io
import mkit.ops.ray
import mkit.ops.tetgen
import numpy as np
import scipy.interpolate
import trimesh
import typer
from numpy import typing as npt


def main(
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
    displacement: Annotated[float, typer.Option()] = 0.05,
) -> None:
    face: trimesh.Trimesh = trimesh.creation.icosphere(radius=0.2)
    pre_skull: trimesh.Trimesh = trimesh.creation.icosphere(radius=0.1)
    post_skull: trimesh.Trimesh = pre_skull.copy()
    post_skull.vertices[post_skull.vertices[:, 1] < 0.0] += [0.0, -displacement, 0.0]
    tri: trimesh.Trimesh = trimesh.util.concatenate([face, pre_skull])
    tet: meshio.Mesh = mkit.ops.tetgen.tetgen(
        mkit.io.as_meshio(
            tri, field_data={"holes": [mkit.ops.ray.find_inner_point(pre_skull)]}
        )
    )
    closest: npt.NDArray[np.floating]
    distance: npt.NDArray[np.floating]
    triangle_id: npt.NDArray[np.integer]
    closest, distance, triangle_id = pre_skull.nearest.on_surface(tet.points)
    mask_skull: npt.NDArray[np.bool_] = distance < 1e-6
    disp: npt.NDArray[np.floating] = np.full(tet.points.shape, np.nan)
    disp[mask_skull] = scipy.interpolate.griddata(
        pre_skull.vertices,
        post_skull.vertices - pre_skull.vertices,
        tet.points[mask_skull],
        method="nearest",
    )
    mkit.io.save(output_file, tet, point_data={"disp": disp})


if __name__ == "__main__":
    mkit.cli.run(main)
