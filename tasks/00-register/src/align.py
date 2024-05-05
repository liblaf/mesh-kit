import pathlib
from typing import Annotated, Optional

import meshio
import mkit.array.mask
import mkit.cli
import mkit.io
import mkit.ops.register.icp
import numpy as np
import trimesh
import typer
from loguru import logger
from numpy import typing as npt


def main(
    source_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    target_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    initial_transform_file: Annotated[
        Optional[pathlib.Path],
        typer.Option("--initial-transform", exists=True, dir_okay=False),
    ] = None,
    output_file: Annotated[
        Optional[pathlib.Path], typer.Option("--output", dir_okay=False, writable=True)
    ],
    output_transform_file: Annotated[
        Optional[pathlib.Path],
        typer.Option("--output-transform", dir_okay=False, writable=True),
    ] = None,
    inverse: Annotated[bool, typer.Option()] = False,
) -> None:
    source_io: meshio.Mesh
    target_io: meshio.Mesh
    source_tr: trimesh.Trimesh
    target_tr: trimesh.Trimesh
    source_io, source_tr = load_mesh_masked(source_file)
    target_io, target_tr = load_mesh_masked(target_file)
    initial: npt.NDArray[np.float64] | None = (
        np.load(initial_transform_file) if initial_transform_file is not None else None
    )
    source_to_target: npt.NDArray[np.float64]
    cost: float
    source_to_target, cost = align(
        source_tr, target_tr, initial=initial, inverse=inverse
    )
    if output_file is not None:
        source_tr.apply_transform(source_to_target)
        source_io.points = source_tr.vertices
        source_io.write(output_file)
    if output_transform_file is not None:
        np.save(output_transform_file, source_to_target)


def load_mesh_masked(
    file: pathlib.Path,
) -> tuple[meshio.Mesh, trimesh.Trimesh]:
    mesh_io: meshio.Mesh = mkit.io.load_meshio(file)
    mesh_tr: trimesh.Trimesh = mkit.io.as_trimesh(mesh_io)
    vert_mask: npt.NDArray[np.bool_] | None = mesh_io.point_data.get("mask", None)
    if vert_mask is not None:
        vert_mask = vert_mask.astype(np.bool_)
        face_mask: npt.NDArray[np.bool_] = mkit.array.mask.vertex_to_face(
            mesh_tr.faces, vert_mask
        )
        mesh_tr.update_faces(face_mask)
        mesh_tr = mesh_tr.process()
    return mesh_io, mesh_tr


def align(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    initial: npt.NDArray[np.float64] | None = None,
    inverse: bool = False,
) -> tuple[npt.NDArray[np.float64], float]:
    if initial is None:
        initial = trimesh.transformations.concatenate_matrices(
            trimesh.transformations.translation_matrix(target.centroid),
            trimesh.transformations.scale_matrix(target.scale),
            trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0]),
            trimesh.transformations.scale_matrix(1.0 / source.scale),
            trimesh.transformations.translation_matrix(-source.centroid),
        )
    source_to_target: npt.NDArray[np.float64]
    cost: float
    source_to_target, cost = mkit.ops.register.icp.icp(
        source, target, initial=initial, inverse=inverse
    )
    logger.info("ICP Cost: {}", cost)
    return source_to_target, cost


if __name__ == "__main__":
    mkit.cli.run(main)
