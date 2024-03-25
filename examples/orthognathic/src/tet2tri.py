import pathlib
from typing import Annotated, no_type_check

import numpy as np
import taichi as ti
import trimesh
import typer
from mkit import cli as _cli
from mkit.io.taichi import mesh as io_ti_mesh
from numpy import typing as npt


@no_type_check
@ti.kernel
def mark_surface(mesh: ti.template()):
    for f in mesh.faces:
        f.faces_ = [f.verts[i].id for i in ti.static(range(3))]
        f.surface = f.cells.size == 1


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    output_file: Annotated[pathlib.Path, typer.Argument(dir_okay=False, writable=True)],
) -> None:
    ti.init()
    mesh_ti: ti.MeshInstance = io_ti_mesh.load(input_file, relations=["FC", "FV"])
    mesh_ti.faces.place(
        {"faces_": ti.types.vector(n=3, dtype=ti.i32), "surface": ti.u1}
    )
    mark_surface(mesh_ti)
    faces_ti: ti.MatrixField = mesh_ti.faces.get_member_field("faces_")
    faces_np: npt.NDArray = faces_ti.to_numpy()
    surface_fi: ti.ScalarField = mesh_ti.faces.get_member_field("surface")
    surface_np: npt.NDArray = surface_fi.to_numpy()
    mesh_tr: trimesh.Trimesh = trimesh.Trimesh(
        vertices=mesh_ti.get_position_as_numpy(), faces=faces_np[np.where(surface_np)]
    )
    mesh_tr = mesh_tr.process(validate=True)
    mesh_tr.export(output_file)


if __name__ == "__main__":
    _cli.run(main)
