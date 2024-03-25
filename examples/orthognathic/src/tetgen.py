import pathlib
from typing import Annotated

import meshio
import numpy as np
import trimesh
import typer
from meshpy import tet
from mkit import cli as _cli
from mkit import points as _points
from mkit.trimesh import ray as _ray
from mkit.typing import as_any as _a
from numpy import typing as npt


def main(
    pre_face_file: Annotated[
        pathlib.Path, typer.Option("--pre-face", exists=True, dir_okay=False)
    ],
    pre_skull_file: Annotated[
        pathlib.Path, typer.Option("--pre-skull", exists=True, dir_okay=False)
    ],
    post_skull_file: Annotated[
        pathlib.Path, typer.Option("--post-skull", exists=True, dir_okay=False)
    ],
    output_file: Annotated[pathlib.Path, typer.Option("--output", dir_okay=False)],
) -> None:
    pre_face: trimesh.Trimesh = _a(trimesh.load(pre_face_file))
    pre_skull: trimesh.Trimesh = _a(trimesh.load(pre_skull_file))
    post_skull: trimesh.Trimesh = _a(trimesh.load(post_skull_file))
    hole: npt.NDArray = _ray.inner_point(pre_skull)
    mesh_tr: trimesh.Trimesh = pre_face.difference(pre_skull)
    mesh_te = tet.MeshInfo()
    mesh_te.set_points(mesh_tr.vertices)
    mesh_te.set_facets(mesh_tr.faces)
    mesh_te.set_holes([hole])
    tetra_te: tet.MeshInfo = tet.build(mesh_te, tet.Options("pYqOzCV"))
    points: npt.NDArray = np.asarray(tetra_te.points)
    elements: npt.NDArray = np.asarray(tetra_te.elements)
    fixed: npt.NDArray = np.full(shape=points.shape, fill_value=np.nan)
    fixed[_points.pos2idx(points, pre_skull.vertices)] = post_skull.vertices
    tetra_io: meshio.Mesh = meshio.Mesh(
        points=points, cells={"tetra": elements}, point_data={"fixed": fixed}
    )
    tetra_io.write(output_file)


if __name__ == "__main__":
    _cli.run(main)
