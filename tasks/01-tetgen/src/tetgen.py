import pathlib
from typing import Annotated

import meshio
import mkit.array.points
import mkit.cli
import mkit.io
import mkit.ops.mesh_fix
import mkit.ops.ray
import mkit.ops.tetgen
import numpy as np
import scipy.interpolate
import trimesh
import typer
from numpy import typing as npt


def main(
    pre_face_file: Annotated[pathlib.Path, typer.Option("--pre-face", exists=True)],
    pre_mandible_file: Annotated[
        pathlib.Path, typer.Option("--pre-mandible", exists=True)
    ],
    pre_maxilla_file: Annotated[
        pathlib.Path, typer.Option("--pre-maxilla", exists=True)
    ],
    post_mandible_file: Annotated[
        pathlib.Path, typer.Option("--post-mandible", exists=True)
    ],
    post_maxilla_file: Annotated[
        pathlib.Path, typer.Option("--post-maxilla", exists=True)
    ],
    *,
    output_file: Annotated[pathlib.Path, typer.Option("-o", "--output", writable=True)],
) -> None:
    pre_face: trimesh.Trimesh = mkit.io.load_trimesh(pre_face_file)
    pre_mandible: trimesh.Trimesh = mkit.io.load_trimesh(pre_mandible_file)
    pre_maxilla: trimesh.Trimesh = mkit.io.load_trimesh(pre_maxilla_file)
    pre_skull: trimesh.Trimesh = trimesh.util.concatenate([pre_mandible, pre_maxilla])
    post_mandible: trimesh.Trimesh = mkit.io.load_trimesh(post_mandible_file)
    post_maxilla: trimesh.Trimesh = mkit.io.load_trimesh(post_maxilla_file)
    post_skull: trimesh.Trimesh = trimesh.util.concatenate(
        [post_mandible, post_maxilla]
    )

    pre_face_repaired: trimesh.Trimesh = mkit.ops.mesh_fix.mesh_fix(pre_face)
    pre_mandible_repaired: trimesh.Trimesh = mkit.ops.mesh_fix.mesh_fix(pre_mandible)
    pre_maxilla_repaired: trimesh.Trimesh = mkit.ops.mesh_fix.mesh_fix(pre_maxilla)
    pre_skull_repaired: trimesh.Trimesh = trimesh.boolean.union(
        [pre_mandible_repaired, pre_maxilla_repaired]
    )
    pre_repaired: trimesh.Trimesh = trimesh.util.concatenate(
        [pre_face_repaired, pre_skull_repaired]
    )
    hole: npt.NDArray[np.floating] = mkit.ops.ray.find_inner_point(pre_skull_repaired)
    tetra: meshio.Mesh = mkit.ops.tetgen.tetgen(
        mkit.io.as_meshio(pre_repaired, field_data={"holes": [hole]})
    )

    closest: npt.NDArray[np.floating]
    distance: npt.NDArray[np.floating]
    triangle_id: npt.NDArray[np.integer]
    closest, distance, triangle_id = pre_skull_repaired.nearest.on_surface(tetra.points)
    skull_mask: npt.NDArray[np.bool_] = distance < 1e-6 * pre_skull.scale
    disp_origin: npt.NDArray[np.floating] = post_skull.vertices - pre_skull.vertices
    disp_interp: npt.NDArray[np.floating] = scipy.interpolate.griddata(
        pre_skull.vertices, disp_origin, tetra.points[skull_mask], method="nearest"
    )
    disp: npt.NDArray[np.floating] = np.full(tetra.points.shape, np.nan)
    disp[skull_mask] = disp_interp
    mkit.io.save(output_file, tetra, point_data={"disp": disp})


if __name__ == "__main__":
    mkit.cli.run(main)
