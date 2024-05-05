import pathlib
from typing import Annotated

import meshio
import mkit.cli
import mkit.io
import mkit.ops.mesh_fix
import mkit.ops.ray
import mkit.ops.tetgen
import numpy as np
import pyvista as pv
import scipy
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
    # post_mandible_file: Annotated[
    #     pathlib.Path, typer.Option("--post-mandible", exists=True)
    # ],
    # post_maxilla_file: Annotated[
    #     pathlib.Path, typer.Option("--post-maxilla", exists=True)
    # ],
    # *,
    # output_file: Annotated[pathlib.Path, typer.Option("-o", "--output", writable=True)],
) -> None:
    pre_face: trimesh.Trimesh = mkit.io.load_trimesh(pre_face_file)
    pre_mandible: trimesh.Trimesh = mkit.io.load_trimesh(pre_mandible_file)
    pre_maxilla: trimesh.Trimesh = mkit.io.load_trimesh(pre_maxilla_file)
    pre_skull: trimesh.Trimesh = trimesh.util.concatenate([pre_mandible, pre_maxilla])
    # post_mandible: trimesh.Trimesh = mkit.io.load_trimesh(post_mandible_file)
    # post_maxilla: trimesh.Trimesh = mkit.io.load_trimesh(post_maxilla_file)
    mkit.io.save("pre-face.ply", pre_face)
    mkit.io.save("pre-skull.ply", trimesh.util.concatenate([pre_mandible, pre_maxilla]))  # pyright: ignore [reportArgumentType]

    pre_face_repaired: trimesh.Trimesh = mkit.ops.mesh_fix.mesh_fix(pre_face)
    pre_mandible_repaired: trimesh.Trimesh = mkit.ops.mesh_fix.mesh_fix(pre_mandible)
    pre_maxilla_repaired: trimesh.Trimesh = mkit.ops.mesh_fix.mesh_fix(pre_maxilla)
    pre_skull_repaired: trimesh.Trimesh = trimesh.boolean.union(
        [pre_mandible_repaired, pre_maxilla_repaired]
    )
    pre_repaired: trimesh.Trimesh = trimesh.util.concatenate(
        [pre_face_repaired, pre_skull_repaired]
    )
    hole: npt.NDArray[np.float64] = mkit.ops.ray.find_inner_point(pre_skull_repaired)
    tetra: meshio.Mesh = mkit.ops.tetgen.tetgen(
        mkit.io.as_meshio(pre_repaired, field_data={"holes": [hole]})
    )
    mkit.io.save("tetra.vtu", tetra)

    disp_origin: npt.NDArray[np.float64] = np.tile(
        np.asarray([100, 0, 0], np.float64), (len(pre_skull.vertices), 1)
    )
    disp_interp: npt.NDArray[np.float64] = scipy.interpolate.griddata(
        pre_skull.vertices, disp_origin, pre_skull_repaired.vertices, method="nearest"
    )
    post_skull: trimesh.Trimesh = pre_skull.copy()
    post_skull.vertices += disp_origin
    post_skull_repaired: trimesh.Trimesh = pre_skull_repaired.copy()
    post_skull_repaired.vertices += disp_interp
    mkit.io.save("post-skull-gt.ply", post_skull)
    mkit.io.save("post-skull-interp.ply", post_skull_repaired)

    pre_skull_pv: pv.PolyData = mkit.io.as_pyvista(pre_skull)
    pre_skull_pv.point_data["disp"] = disp_origin
    pre_skull_repaired_pv: pv.PolyData = mkit.io.as_pyvista(pre_skull_repaired)
    pre_skull_repaired_pv = pre_skull_repaired_pv.sample(pre_skull_pv)
    disp_interp = pre_skull_repaired_pv.point_data["disp"]
    post_skull_repaired = pre_skull_repaired.copy()
    post_skull_repaired.vertices += disp_interp
    mkit.io.save("post-skull-naive.ply", post_skull_repaired)


if __name__ == "__main__":
    mkit.cli.run(main)
