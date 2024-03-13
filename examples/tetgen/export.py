import pathlib
from typing import Annotated, no_type_check

import meshtaichi_patcher
import numpy as np
import taichi as ti
import trimesh
import typer
from mesh_kit import cli
from numpy import typing as npt


@no_type_check
@ti.kernel
def get_faces_all(mesh: ti.template(), faces: ti.template()):
    for f in mesh.faces:
        faces[f.id] = [f.verts[i].id for i in ti.static(range(3))]


@no_type_check
@ti.kernel
def get_faces(mesh: ti.template(), faces: ti.template()):
    for f in mesh.faces:
        if f.cells.size == 1:
            faces[f.id] = [f.verts[i].id for i in ti.static(range(3))]


def main(
    file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
    interior: Annotated[bool, typer.Option()] = True,
) -> None:
    ti.init(debug=True)
    mesh: ti.MeshInstance = meshtaichi_patcher.load_mesh(str(file), ["FC", "FV"])
    faces: ti.MatrixField = ti.Vector.field(3, int, shape=(len(mesh.faces),))
    faces.fill(-1)
    if interior:
        get_faces_all(mesh, faces)
    else:
        get_faces(mesh, faces)
    faces_numpy: npt.NDArray = faces.to_numpy()
    output = trimesh.Trimesh(
        mesh.get_position_as_numpy(), faces_numpy[np.all(faces_numpy >= 0, axis=1)]
    )
    output.export(output_file)


if __name__ == "__main__":
    cli.run(main)
