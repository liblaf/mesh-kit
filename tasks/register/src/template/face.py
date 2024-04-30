import pathlib
from typing import Annotated

import mkit.array.mask
import mkit.cli
import mkit.ops.mesh_fix
import numpy as np
import trimesh
import trimesh.bounds
import typer
from mkit import io as _io
from numpy import typing as npt


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True)],
    output_file: Annotated[pathlib.Path, typer.Option("-o", "--output", writable=True)],
) -> None:
    mesh: trimesh.Trimesh = _io.load_trimesh(input_file)
    vertex_mask: npt.NDArray[np.bool_] = mesh.vertices[:, 2] > -50
    vertex_mask &= ~(
        trimesh.bounds.contains([[-25, -np.inf, 10], [30, 100, 15]], mesh.vertices)
        & (mesh.vertex_normals[:, 1] > -0.6)
    )
    face_mask: npt.NDArray[np.bool_] = mkit.array.mask.vertex_to_face(
        mesh.faces, vertex_mask
    )
    mesh.update_faces(face_mask)
    mesh = mkit.ops.mesh_fix.mesh_fix(mesh, verbose=True)
    _io.save(output_file, mesh)


if __name__ == "__main__":
    mkit.cli.run(main)
