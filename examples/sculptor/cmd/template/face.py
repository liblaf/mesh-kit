import pathlib
from typing import Annotated

import numpy as np
import trimesh
import typer
from numpy import typing as npt
from trimesh import bounds

from mesh_kit.common import cli, testing


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    output_file: Annotated[pathlib.Path, typer.Argument(dir_okay=False, writable=True)],
) -> None:
    mesh: trimesh.Trimesh = trimesh.load(input_file)
    vertex_mask: npt.NDArray = mesh.vertices[:, 2] > -50
    testing.assert_shape(vertex_mask.shape, (mesh.vertices.shape[0],))
    vertex_mask &= ~bounds.contains(
        bounds=[[-25, -np.inf, 10], [30, 100, 15]], points=mesh.vertices
    )
    face_mask: npt.NDArray = vertex_mask[mesh.faces].all(axis=1)
    testing.assert_shape(face_mask.shape, (mesh.faces.shape[0],))
    mesh.update_faces(face_mask)
    mesh.export(output_file)


if __name__ == "__main__":
    cli.run(main)
