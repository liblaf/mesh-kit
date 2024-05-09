import pathlib
from typing import Annotated

import meshio
import mkit.array.points
import mkit.cli
import mkit.io
import mkit.ops.mesh_fix
import mkit.ops.ray
import mkit.ops.tetgen
import mkit.tetra
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
    pre_face: meshio.Mesh = mkit.io.load_meshio(pre_face_file)
    pre_mandible: meshio.Mesh = mkit.io.load_meshio(pre_mandible_file)
    pre_maxilla: meshio.Mesh = mkit.io.load_meshio(pre_maxilla_file)
    post_mandible: meshio.Mesh = mkit.io.load_meshio(post_mandible_file)
    post_maxilla: meshio.Mesh = mkit.io.load_meshio(post_maxilla_file)
    tetra: meshio.Mesh = tetgen(
        face=mkit.io.as_trimesh(pre_face),
        mandible=mkit.io.as_trimesh(pre_mandible),
        maxilla=mkit.io.as_trimesh(pre_maxilla),
    )
    face_mask: npt.NDArray[np.bool_]
    skull_mask: npt.NDArray[np.bool_]
    face_mask, skull_mask = extract_face_skull(tetra)

    disp: npt.NDArray[np.floating] = np.full(tetra.points.shape, np.nan)
    disp[skull_mask] = scipy.interpolate.griddata(
        np.vstack((pre_mandible.points, pre_maxilla.points)),
        np.vstack(
            (
                post_mandible.points - pre_mandible.points,
                post_maxilla.points - pre_maxilla.points,
            )
        ),
        tetra.points[skull_mask],
    )
    validation: npt.NDArray[np.bool_] = np.full((len(tetra.points),), False)
    validation[face_mask] = scipy.interpolate.griddata(
        pre_face.points, pre_face.point_data["validation"], tetra.points[face_mask]
    )

    mkit.io.save(
        output_file,
        tetra,
        point_data={"disp": disp, "validation": validation.astype(np.int8)},
    )


def tetgen(
    face: trimesh.Trimesh,
    mandible: trimesh.Trimesh,
    maxilla: trimesh.Trimesh,
) -> meshio.Mesh:
    face_repaired: trimesh.Trimesh = mkit.ops.mesh_fix.mesh_fix(face)
    mandible_repaired: trimesh.Trimesh = mkit.ops.mesh_fix.mesh_fix(mandible)
    maxilla_repaired: trimesh.Trimesh = mkit.ops.mesh_fix.mesh_fix(maxilla)
    skull_repaired: trimesh.Trimesh = trimesh.boolean.union(
        [mandible_repaired, maxilla_repaired]
    )
    repaired: trimesh.Trimesh = trimesh.util.concatenate(
        [face_repaired, skull_repaired]
    )
    hole: npt.NDArray[np.floating] = mkit.ops.ray.find_inner_point(skull_repaired)
    tetra: meshio.Mesh = mkit.ops.tetgen.tetgen(
        mkit.io.as_meshio(repaired, field_data={"holes": [hole]})
    )
    return tetra


def extract_face_skull(
    tetra: meshio.Mesh,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    boundary_faces: npt.NDArray[np.integer] = mkit.tetra.boundary_faces(
        tetra.get_cells_type("tetra")
    )
    triangle = trimesh.Trimesh(tetra.points, boundary_faces)
    face: trimesh.Trimesh
    skull: trimesh.Trimesh
    face, skull = triangle.split()
    if np.prod(face.extents) < np.prod(skull.extents):
        face, skull = skull, face
    face_mask: npt.NDArray[np.bool_] = mkit.array.points.position_to_mask(
        tetra.points, face.vertices
    )
    skull_mask: npt.NDArray[np.bool_] = mkit.array.points.position_to_mask(
        tetra.points, skull.vertices
    )
    return face_mask, skull_mask


if __name__ == "__main__":
    mkit.cli.run(main)
