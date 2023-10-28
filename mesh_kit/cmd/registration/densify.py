from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pytorch3d.ops
import torch
from numpy.typing import NDArray
from pytorch3d.io import IO
from pytorch3d.ops.points_alignment import SimilarityTransform
from pytorch3d.structures import Meshes
from torch import Tensor
from typer import Argument, Option

from mesh_kit.common.cli import run


def main(
    source_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    target_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    *,
    device: Annotated[str, Option()] = "cuda",
    neighbor_radius: Annotated[float, Option()] = 10.0,
    num_landmarks: Annotated[int, Option()] = 10000,
    num_neighbors: Annotated[int, Option()] = 500,
) -> None:
    io: IO = IO()
    source: Meshes = io.load_mesh(path=source_filepath, device=device)
    target: Meshes = io.load_mesh(path=target_filepath, device=device)
    # (num_landmarks_init, 3)
    source_landmarks: NDArray = np.loadtxt(
        fname=source_filepath.with_suffix(".landmark.txt")
    )
    # (num_landmarks_init, 3)
    target_landmarks: NDArray = np.loadtxt(
        fname=target_filepath.with_suffix(".landmark.txt")
    )
    transform: SimilarityTransform = pytorch3d.ops.corresponding_points_alignment(
        torch.from_numpy(source_landmarks).to(device=device).unsqueeze(dim=0),
        torch.from_numpy(target_landmarks).to(device=device).unsqueeze(dim=0),
        estimate_scale=True,
    )
    # (1, num_source_verts, 3)
    source_verts: Tensor = (
        transform.s[:, None, None]
        * torch.bmm(transform.R, cast(Tensor, source.verts_padded()))
        + transform.T[:, None, :]
    )
    source = Meshes(verts=source_verts, faces=source.faces_padded())
    samples: Tensor  # (1, num_landmarks, 3)
    normals: Tensor  # (1, num_landmarks, 3)
    samples, normals = cast(
        tuple[Tensor, Tensor],
        pytorch3d.ops.sample_points_from_meshes(
            meshes=source,
            num_samples=num_landmarks,
            return_normals=True,
            return_textures=False,
        ),
    )
    dists: Tensor  # (1, num_landmarks, num_neighbors)
    idx: Tensor  # (1, num_landmarks, num_neighbors)
    nn: Tensor  # (1, num_landmarks, num_neighbors, 3)
    dists, idx, nn = pytorch3d.ops.ball_query(
        samples,
        cast(Tensor, target.verts_padded()),
        K=num_neighbors,
        radius=neighbor_radius,
        return_nn=True,
    )
    # (1, num_landmarks, num_neighbors, 3)
    target_verts_normals: Tensor = pytorch3d.ops.knn_gather(
        cast(Tensor, target.verts_normals_padded()), idx=idx
    )
    # (1, num_landmarks, num_neighbors)
    loss: Tensor = dists + torch.einsum("NLU,NLKU->NLK", normals, target_verts_normals)
    # (1, num_landmarks)
    landmark_idx: Tensor = torch.argmin(loss, dim=-1)
    target_landmarks_dense: Tensor = pytorch3d.ops.knn_gather(
        cast(Tensor, target.verts_padded()), idx=landmark_idx.unsqueeze(dim=-1)
    )


if __name__ == "__main__":
    run(main)
