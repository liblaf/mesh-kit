import pathlib
from typing import Annotated

import meshio
import numpy as np
import trimesh
import typer
from mkit import io as _io
from mkit import points
from mkit import tetgen as _tetgen
from mkit.trimesh import ray as _ray
from numpy import typing as npt


def main(
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
) -> None:
    pre_face: trimesh.Trimesh = trimesh.load(
        "/home/liblaf/Documents/data/template/04-face.ply"
    )
    # pre_face = pre_face.subdivide()
    pre_skull: trimesh.Trimesh = trimesh.load(
        "/home/liblaf/Documents/data/template/04-skull.ply"
    )
    post_skull: trimesh.Trimesh = pre_skull.copy()
    vert_mask: npt.NDArray = post_skull.vertices[:, 2] < -20.0
    post_skull.vertices[vert_mask] += [0, -10.0, 0]
    post_skull.export("post-skull.ply")

    mesh_tr: trimesh.Trimesh = pre_face.difference(pre_skull)
    tetgen: _tetgen.TetGen = _tetgen.from_meshio(_io.from_trimesh(mesh_tr))
    tetgen.field_data["holes"] = np.asarray([_ray.inner_point(pre_skull)])
    tetra_io: meshio.Mesh = tetgen.tetgen()

    fixed_mask: npt.NDArray = points.pos2idx(tetra_io.points, pre_skull.vertices)
    disp: npt.NDArray = np.full(shape=(len(tetra_io.points), 3), fill_value=np.nan)
    disp[fixed_mask] = post_skull.vertices - pre_skull.vertices
    tetra_io.point_data["disp"] = disp
    tetra_io.write(output_file)


if __name__ == "__main__":
    typer.run(main)
