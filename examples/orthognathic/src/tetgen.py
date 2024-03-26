import pathlib
import subprocess
from typing import Annotated, Any

import meshio
import numpy as np
import trimesh
import typer
from mkit import cli as _cli
from mkit import points as _points
from mkit.io.tetgen import ele as io_ele
from mkit.io.tetgen import node as io_node
from mkit.io.tetgen import smesh as io_smesh
from mkit.trimesh import ray as _ray
from mkit.typing import as_any as _a
from numpy import typing as npt


def tetgen(mesh: trimesh.Trimesh, holes: npt.NDArray) -> meshio.Mesh:
    _: Any
    io_smesh.save(
        pathlib.Path("tetgen.smesh"),
        verts=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        holes=np.asarray(holes),
    )
    subprocess.run(
        [
            "tetgen",
            "-p",
            # "-Y",
            "-q",
            "-a0.00001",
            "-O",
            "-z",
            "-C",
            "-V",
            "tetgen.smesh",
        ],
        check=True,
    )
    points: npt.NDArray
    points, _, _ = io_node.load(pathlib.Path("tetgen.1.node"))
    elements: npt.NDArray
    elements, _ = io_ele.load(pathlib.Path("tetgen.1.ele"))
    return meshio.Mesh(points=points, cells={"tetra": elements})


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
    # mesh_te = tet.MeshInfo()
    # mesh_te.set_points(mesh_tr.vertices)
    # mesh_te.set_facets(mesh_tr.faces)
    # mesh_te.set_holes([hole])
    # tetra_te: tet.MeshInfo = tet.build(mesh_te, tet.Options("pYq1.414zCV"))
    tetra_io: meshio.Mesh = tetgen(mesh_tr, np.asarray([hole]))
    points = tetra_io.points
    disp: npt.NDArray = np.full(shape=points.shape, fill_value=np.nan)
    fixed_mask: npt.NDArray = _points.pos2idx(points, pre_skull.vertices)
    disp[fixed_mask] = post_skull.vertices - pre_skull.vertices
    tetra_io.point_data["disp"] = disp
    tetra_io.write(output_file)


if __name__ == "__main__":
    _cli.run(main)
