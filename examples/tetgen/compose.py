import pathlib
from typing import Annotated, Any

import numpy as np
import trimesh
import typer
from mesh_kit import cli, points
from mesh_kit.io import smesh
from mesh_kit.typing import check_type as _t
from numpy import typing as npt


def find_inner_point(mesh: trimesh.Trimesh) -> npt.NDArray:
    _: Any
    rng: np.random.Generator = np.random.default_rng()
    idx: int = rng.integers(0, len(mesh.vertices))
    idx = 0
    origin: npt.NDArray = mesh.vertices[idx]
    direction: npt.NDArray = -mesh.vertex_normals[idx]
    locations: npt.NDArray
    locations, _, _ = mesh.ray.intersects_location(
        [origin], [direction], multiple_hits=True
    )
    return locations[:2].mean(axis=0)


def main(
    face_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    skull_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
) -> None:
    face: trimesh.Trimesh = _t(trimesh.Trimesh, trimesh.load(face_file))
    skull: trimesh.Trimesh = _t(trimesh.Trimesh, trimesh.load(skull_file))
    mesh: trimesh.Trimesh = face.difference(skull)
    attrs: npt.NDArray = np.full((len(mesh.vertices), 1), fill_value=-1, dtype=int)
    face_vert_idx: npt.NDArray = points.position2idx(mesh.vertices, face.vertices)
    attrs[face_vert_idx] = 1
    skull_vert_idx: npt.NDArray = points.position2idx(mesh.vertices, skull.vertices)
    attrs[skull_vert_idx] = 2
    hole: npt.NDArray = find_inner_point(skull)
    smesh.save(
        output_file, mesh.vertices, mesh.faces, holes=np.asarray([hole]), attrs=attrs
    )


if __name__ == "__main__":
    cli.run(main)
