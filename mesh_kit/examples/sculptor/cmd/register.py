import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Optional, cast

import numpy as np
import tqdm.rich
import trimesh
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from trimesh import Trimesh
from typer import Argument, Option

from mesh_kit.common.cli import run
from mesh_kit.common.path import landmarks_filepath


def select_landmark(
    source_idx: int,
    target_idx: Sequence[int],
    *,
    source: Trimesh,
    target: Trimesh,
    normal_weight: float = 1.0,
) -> tuple[int, float]:
    distance_loss: NDArray = np.linalg.norm(
        target.vertices[target_idx, :] - source.vertices[source_idx, :], axis=1
    )
    normal_loss: NDArray = 1.0 - np.dot(
        target.vertex_normals[target_idx, :], source.vertex_normals[source_idx, :]
    )
    loss: NDArray = distance_loss + normal_weight * normal_loss
    arg_min: int = cast(int, np.argmin(loss))
    return target_idx[arg_min], loss[arg_min]


def select_landmarks(
    source: Trimesh,
    target: Trimesh,
    normal_weight: float = 1.0,
    threshold: float = 1.0,
) -> tuple[NDArray, NDArray]:
    rng: np.random.Generator = np.random.default_rng()
    source_landmarks: NDArray = rng.choice(
        source.vertices.shape[0], size=(source.vertices.shape[0],), replace=False
    )
    source_tree: cKDTree = cKDTree(data=source.vertices[source_landmarks, :])
    target_tree: cKDTree = target.kdtree
    target_landmarks: NDArray = np.zeros_like(source_landmarks)
    loss: NDArray = np.zeros_like(source_landmarks, dtype=float)
    for i, target_idx in enumerate(
        source_tree.query_ball_tree(other=target_tree, r=source.scale / 20.0, p=2.0),
    ):
        source_idx: int = source_landmarks[i]
        target_idx: list[int] = target_idx
        target_landmarks[i], loss[i] = select_landmark(
            source_idx=source_idx,
            target_idx=target_idx,
            source=source,
            target=target,
            normal_weight=normal_weight,
        )
    idx: NDArray = (source.scale / 100.0 < loss) & (loss < source.scale / 50.0)
    source_landmarks = source_landmarks[idx]
    target_landmarks = target_landmarks[idx]
    logging.info(f"num of selected landmarks: {source_landmarks.shape[0]}")
    return source_landmarks, target.vertices[target_landmarks, :]


def main(
    source_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    target_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    *,
    output_filepath: Annotated[Path, Option("--output", dir_okay=False, writable=True)],
    records_filepath: Annotated[
        Optional[Path], Option("--records", exists=True, file_okay=False, writable=True)
    ] = None,
    normal_weight: Annotated[float, Option(min=0.0)] = 1.0,
    num_iters: Annotated[int, Option(min=0)] = 10,
    threshold: Annotated[float, Option(min=0.0)] = 1.0,
) -> None:
    source: Trimesh = cast(Trimesh, trimesh.load(source_filepath))
    target: Trimesh = cast(Trimesh, trimesh.load(target_filepath))
    source_positions_input: NDArray = np.loadtxt(landmarks_filepath(source_filepath))
    source_landmarks_input: NDArray
    _, source_landmarks_input = source.nearest.vertex(points=source_positions_input)
    target_positions_input: NDArray = np.loadtxt(landmarks_filepath(target_filepath))
    source_landmarks: NDArray
    target_positions: NDArray
    for i in tqdm.rich.trrange(num_iters):
        if i % 2 == 0:
            source_landmarks = source_landmarks_input
            target_positions = target_positions_input
        else:
            source_landmarks, target_positions = select_landmarks(
                source=source,
                target=target,
                normal_weight=normal_weight,
                threshold=threshold,
            )
        if records_filepath is not None:
            source.export(records_filepath / f"{i:02d}.ply")
            np.savetxt(
                fname=records_filepath / f"{i:02d}-source-landmarks.txt",
                X=source.vertices[source_landmarks, :],
            )
            np.savetxt(
                fname=records_filepath / f"{i:02d}-target-landmarks.txt",
                X=target_positions,
            )
        result: NDArray = cast(
            NDArray,
            trimesh.registration.nricp_amberg(
                source_mesh=source,
                target_geometry=target,
                source_landmarks=source_landmarks,
                target_positions=target_positions,
                # smoothness, landmark, normal, max_iter
                steps=[
                    [0.01, 10, 0.2, 10],
                    [0.02, 5, 0.2, 10],
                    [0.03, 2.5, 0.2, 10],
                    [0.01, 0, 0.0, 10],
                ],
                distance_threshold=source.scale / 100,
            ),
        )
        source = Trimesh(vertices=result, faces=source.faces)
    source.export(output_filepath)


if __name__ == "__main__":
    run(main)
