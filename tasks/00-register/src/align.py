import pathlib
from typing import Annotated, Optional

import meshio
import mkit.array.mask
import mkit.array.points
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
    source_landmarks_file: Annotated[
        Optional[pathlib.Path], typer.Argument(exists=True, dir_okay=False)
    ] = None,
    target_landmarks_file: Annotated[
        Optional[pathlib.Path], typer.Argument(exists=True, dir_okay=False)
    ] = None,
    initial_transform_file: Annotated[
        Optional[pathlib.Path],
        typer.Option("--initial-transform", exists=True, dir_okay=False),
    ] = None,
    output_file: Annotated[
        Optional[pathlib.Path], typer.Option("--output", dir_okay=False, writable=True)
    ] = None,
    output_transform_file: Annotated[
        Optional[pathlib.Path],
        typer.Option("--output-transform", dir_okay=False, writable=True),
    ] = None,
    inverse: Annotated[bool, typer.Option()] = False,
    smart_initial: Annotated[bool, typer.Option()] = False,
) -> None:
    mkit.cli.up_to_date(
        [output_file, output_transform_file],
        [
            __file__,
            source_file,
            target_file,
            source_landmarks_file,
            target_landmarks_file,
            initial_transform_file,
        ],
    )
    source: meshio.Mesh
    target: meshio.Mesh
    source_masked: trimesh.Trimesh
    target_masked: trimesh.Trimesh
    source_landmarks: npt.NDArray[np.integer]
    target_landmarks: npt.NDArray[np.integer]
    source, source_masked, source_landmarks = load_mesh_masked(
        source_file, source_landmarks_file
    )
    target, target_masked, target_landmarks = load_mesh_masked(
        target_file, target_landmarks_file
    )
    initial: npt.NDArray[np.floating] | None = None
    cost: float
    if initial_transform_file:
        initial = np.load(initial_transform_file)
    elif len(source_landmarks) > 5:
        initial, cost = trimesh.registration.procrustes(
            source_landmarks, target_landmarks
        )
        logger.info("Procrustes Cost: {}", cost)
    source_to_target: npt.NDArray[np.floating]
    source_to_target, cost = align(
        source_masked,
        target_masked,
        initial=initial,
        inverse=inverse,
        smart_initial=smart_initial,
    )
    if output_file is not None:
        source.points = trimesh.transform_points(source.points, source_to_target)
        mkit.io.save(output_file, source)
    if output_transform_file is not None:
        np.save(output_transform_file, source_to_target)


def load_mesh_masked(
    file: pathlib.Path, landmarks_file: pathlib.Path | None
) -> tuple[meshio.Mesh, trimesh.Trimesh, npt.NDArray[np.integer]]:
    mesh_io: meshio.Mesh = mkit.io.load_meshio(file)
    mesh_tr: trimesh.Trimesh = mkit.io.as_trimesh(mesh_io)
    vert_mask: npt.NDArray[np.bool_] | None = mesh_io.point_data.get("register", None)
    if vert_mask is not None:
        vert_mask = vert_mask.astype(np.bool_)
        face_mask: npt.NDArray[np.bool_] = mkit.array.mask.vertex_to_face(
            mesh_tr.faces, vert_mask
        )
        mesh_tr.update_faces(face_mask)
        mesh_tr = mesh_tr.process()
    landmarks: npt.NDArray[np.integer] = load_landmarks(mesh_io, landmarks_file)
    return mesh_io, mesh_tr, landmarks


def load_landmarks(
    mesh: meshio.Mesh, landmarks_file: pathlib.Path | None
) -> npt.NDArray[np.integer]:
    if "landmarks" in mesh.field_data:
        return mesh.field_data["landmarks"]
    if landmarks_file is None or not landmarks_file.exists():
        return np.empty((0, 3), np.intp)
    points: npt.NDArray[np.floating] = np.loadtxt(landmarks_file)
    idx: npt.NDArray[np.integer] = mkit.array.points.position_to_index(
        mesh.points, points
    )
    return idx


def align(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    initial: npt.NDArray[np.floating] | None = None,
    inverse: bool = False,
    smart_initial: bool = False,
) -> tuple[npt.NDArray[np.floating], float]:
    if initial is None and smart_initial:
        initial = trimesh.transformations.concatenate_matrices(
            trimesh.transformations.translation_matrix(target.centroid),
            trimesh.transformations.scale_matrix(target.scale),
            trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0]),
            trimesh.transformations.scale_matrix(1.0 / source.scale),
            trimesh.transformations.translation_matrix(-source.centroid),
        )
    source_to_target: npt.NDArray[np.floating]
    cost: float
    source_to_target, cost = mkit.ops.register.icp.icp(
        source, target, initial=initial, inverse=inverse
    )
    logger.info("ICP Cost: {}", cost)
    return source_to_target, cost


if __name__ == "__main__":
    mkit.cli.run(main)
