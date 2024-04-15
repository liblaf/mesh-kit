import pathlib
from typing import Any

import meshio
import numpy as np
import trimesh
import typer
from loguru import logger
from mkit import io as _io
from mkit import points
from mkit import tetgen as _tetgen
from mkit.trimesh import ray as _ray
from mkit.typing import as_any as _a
from numpy import typing as npt
from trimesh import registration

DATA_DIR = pathlib.Path("/home/liblaf/Documents/data/targets/赵颖")


def main() -> None:
    _: Any
    pre_face: trimesh.Trimesh = _a(trimesh.load(DATA_DIR / "pre" / "05-face.ply"))
    pre_skull: trimesh.Trimesh = _a(trimesh.load(DATA_DIR / "pre" / "05-skull.ply"))
    post_face: trimesh.Trimesh = _a(trimesh.load(DATA_DIR / "post" / "05-face.ply"))
    post_skull: trimesh.Trimesh = _a(trimesh.load(DATA_DIR / "post" / "05-skull.ply"))
    matrix: npt.NDArray
    cost: float
    matrix, _, cost = registration.procrustes(post_skull.vertices, pre_skull.vertices)
    logger.info("procrustes(): cost = {}", cost)
    post_face.apply_transform(matrix)
    post_skull.apply_transform(matrix)
    pre_face.export("data/pre-face.ply")
    pre_skull.export("data/pre-skull.ply")
    post_face.export("data/post-face-gt.ply")
    post_skull.export("data/post-skull.ply")

    mesh_tr: trimesh.Trimesh = pre_face.difference(pre_skull)
    tetgen: _tetgen.TetGen = _tetgen.from_meshio(_io.from_trimesh(mesh_tr))
    hole: npt.NDArray = _ray.inner_point(pre_skull)
    tetgen.field_data["holes"] = np.asarray([hole], dtype=float)
    tetra_io: meshio.Mesh = tetgen.tetgen()

    fixed_mask: npt.NDArray = points.pos2idx(tetra_io.points, pre_skull.vertices)
    disp: npt.NDArray = np.full(shape=(len(tetra_io.points), 3), fill_value=np.nan)
    disp[fixed_mask] = post_skull.vertices - pre_skull.vertices
    tetra_io.point_data["disp"] = disp
    tetra_io.write("patient.vtu")


if __name__ == "__main__":
    typer.run(main)
