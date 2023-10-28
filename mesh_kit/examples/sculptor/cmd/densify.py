from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pytorch3d.ops
import torch
from numpy.random import Generator
from pytorch3d.io import IO
from pytorch3d.structures import Meshes
from torch import Tensor
from typer import Argument, Option

from mesh_kit.common.cli import run


def main(
    source_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    target_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    *,
    device: Annotated[str, Option()] = "cuda",
    output_source_landmarks_path: Annotated[
        Path, Option("--output-source-landmarks", dir_okay=False, writable=True)
    ],
    output_target_landmarks_path: Annotated[
        Path, Option("--output-target-landmarks", dir_okay=False, writable=True)
    ],
    neighbor_radius: Annotated[float, Option()] = 10.0,
    num_landmarks: Annotated[int, Option()] = 10000,
    num_neighbors: Annotated[int, Option()] = 500,
) -> None:
    io: IO = IO()
    rng: Generator = np.random.default_rng()
    torch.set_default_device(device=device)
    source: Meshes = io.load_mesh(path=source_filepath, device=device)
    target: Meshes = io.load_mesh(path=target_filepath, device=device)
    source_landmarks_idx: Tensor = torch.from_numpy(
        rng.choice(
            cast(Tensor, source.num_verts_per_mesh())[0],
            size=(num_landmarks,),
            replace=False,
        )
    ).to(device=device)
    source_landmarks: Tensor = cast(Tensor, source.verts_padded())[
        0, source_landmarks_idx, :
    ].reshape(1, num_landmarks, 3)
    source_landmarks_normals: Tensor = cast(Tensor, source.verts_normals_padded())[
        0, source_landmarks_idx, :
    ].reshape(1, num_landmarks, 3)
    dists: Tensor  # (1, num_landmarks, num_neighbors)
    idx: Tensor  # (1, num_landmarks, num_neighbors)
    nn: Tensor  # (1, num_landmarks, num_neighbors, 3)
    dists, idx, nn = pytorch3d.ops.ball_query(
        source_landmarks,
        cast(Tensor, target.verts_padded()),
        K=num_neighbors,
        radius=neighbor_radius,
        return_nn=True,
    )
    # (1, num_landmarks, num_neighbors, 3)
    target_landmarks_normals: Tensor = pytorch3d.ops.knn_gather(
        cast(Tensor, target.verts_normals_padded()), idx=idx
    )
    loss: Tensor = dists + torch.cosine_similarity(
        source_landmarks_normals.view(num_landmarks, 3),
        target_landmarks_normals.view(num_landmarks * num_neighbors, 3),
    ).view(1, num_landmarks, num_neighbors)
    # (1, num_landmarks)
    target_landmarks_idx: Tensor = torch.argmin(loss, dim=-1)
    # (1, num_landmarks, 3)
    target_landmarks: Tensor = pytorch3d.ops.knn_gather(
        cast(Tensor, target.verts_padded()), idx=target_landmarks_idx.unsqueeze(dim=-1)
    )
    np.savetxt(output_source_landmarks_path, source_landmarks)
    np.savetxt(output_target_landmarks_path, target_landmarks)


if __name__ == "__main__":
    run(main)
