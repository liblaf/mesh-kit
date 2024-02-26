import pathlib
from typing import Annotated

import numpy as np
import trimesh
import typer
from numpy import typing as npt
from trimesh import bounds, smoothing

from mesh_kit import cli
from mesh_kit import trimesh as _tri
from mesh_kit.io import trimesh as _io
from mesh_kit.typing import check_shape as _check_shape


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    output_file: Annotated[pathlib.Path, typer.Argument(dir_okay=False, writable=True)],
    *,
    smooth: Annotated[bool, typer.Option()] = False,
) -> None:
    mesh: trimesh.Trimesh
    attrs: dict[str, npt.NDArray]
    mesh, attrs = _io.read(input_file, attr=True)
    vertex_mask: npt.NDArray = _check_shape(
        mesh.vertices[:, 2] > -50,  # noqa: PLR2004
        (mesh.vertices.shape[0],),
    )
    vertex_mask &= ~bounds.contains(
        bounds=[[-25, -np.inf, 10], [30, 100, 15]], points=mesh.vertices
    )
    face_mask: npt.NDArray = _check_shape(
        _tri.mask.vert2face(mesh, vertex_mask), (mesh.faces.shape[0],)
    )
    mesh.update_faces(face_mask)
    mesh = _tri.mesh_fix(mesh)
    if smooth:
        mesh = smoothing.filter_laplacian(mesh)
    _io.write(output_file, mesh, attr=True, **attrs)


if __name__ == "__main__":
    cli.run(main)
