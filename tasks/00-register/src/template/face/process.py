import pathlib
from typing import Annotated

import mkit.array.mask
import mkit.cli
import mkit.io
import mkit.ops.mesh_fix
import numpy as np
import trimesh
import trimesh.bounds
import typer
from numpy import typing as npt


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
) -> None:
    mkit.cli.up_to_date(output_file, [__file__, input_file])
    mesh: trimesh.Trimesh = mkit.io.load_trimesh(input_file)
    plane_origin: npt.ArrayLike = [-145.73205847, 120.19102723, -46.35020492]
    plane_normal: npt.ArrayLike = [-0.00281506, -0.12219853, 0.99250168]
    mesh = mesh.slice_plane(plane_origin, plane_normal)
    verts_to_remove: npt.NDArray[np.bool_] = trimesh.bounds.contains(
        [[-25.0, -np.inf, 10.0], [30.0, 100.0, 20.0]], mesh.vertices
    )
    # verts_to_remove &= mesh.vertex_normals[:, 2] > -0.8
    faces_to_remove: npt.NDArray[np.bool_] = mkit.array.mask.vertex_to_face(
        mesh.faces, verts_to_remove
    )
    mesh.update_faces(~faces_to_remove)
    mesh = mkit.ops.mesh_fix.mesh_fix(mesh, verbose=True)
    validation_mask: npt.NDArray[np.bool_] = trimesh.bounds.contains(
        [[-np.inf, -np.inf, -40.0], [np.inf, 130.0, 80.0]], mesh.vertices
    ) | trimesh.bounds.contains(
        [[-np.inf, -np.inf, -30.0], [np.inf, 160.0, 80.0]], mesh.vertices
    )
    mkit.io.save(
        output_file,
        mesh,
        point_data={
            "register": np.ones((len(mesh.vertices),), np.int8),
            "validation": validation_mask.astype(np.int8),
        },
    )


if __name__ == "__main__":
    mkit.cli.run(main)
