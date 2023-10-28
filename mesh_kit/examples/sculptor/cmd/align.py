from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pytorch3d.io
import pytorch3d.ops
import torch
from pytorch3d.io import IO
from pytorch3d.ops.points_alignment import ICPSolution, SimilarityTransform
from pytorch3d.structures import Meshes
from torch import Tensor
from typer import Argument, Option

import mesh_kit.transform.similarity
from mesh_kit.common.cli import run


def main(
    source_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    target_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    *,
    device: Annotated[str, Option()] = "cuda",
    output_path: Annotated[Path, Option("--output", dir_okay=False, writable=True)],
    output_landmarks_path: Annotated[
        Path, Option("--output-landmarks", dir_okay=False, writable=True)
    ],
    source_landmarks_path: Annotated[
        Path, Option("--source-landmarks", exists=True, dir_okay=False)
    ],
    target_landmarks_path: Annotated[
        Path, Option("--target-landmarks", exists=True, dir_okay=False)
    ],
) -> None:
    io: IO = IO()
    source: Meshes = io.load_mesh(source_filepath, device=device)
    target: Meshes = io.load_mesh(target_filepath, device=device)
    source_landmarks: Tensor = (
        torch.from_numpy(np.loadtxt(source_landmarks_path, dtype=np.float32))
        .to(device=device)
        .unsqueeze(0)
    )
    target_landmarks: Tensor = (
        torch.from_numpy(np.loadtxt(target_landmarks_path, dtype=np.float32))
        .to(device=device)
        .unsqueeze(0)
    )
    init_transform: SimilarityTransform = pytorch3d.ops.corresponding_points_alignment(
        source_landmarks, target_landmarks, estimate_scale=True, allow_reflection=False
    )
    solution: ICPSolution = pytorch3d.ops.iterative_closest_point(
        cast(Tensor, source.verts_padded()),
        cast(Tensor, target.verts_padded()),
        init_transform=init_transform,
        estimate_scale=True,
        allow_reflection=False,
        verbose=True,
    )
    output = Meshes(verts=solution.Xt, faces=source.faces_padded())
    io.save_mesh(output, output_path, binary=True, include_textures=False)
    output_landmarks: Tensor = mesh_kit.transform.similarity.apply(
        source_landmarks, transform=solution.RTs
    )
    np.savetxt(output_landmarks_path, output_landmarks.numpy())


if __name__ == "__main__":
    run(main)
