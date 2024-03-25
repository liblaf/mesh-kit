import pathlib

import numpy as np
import torch
from numpy.lib import npyio
from pytorch3d.ops import points_alignment


def save(
    file: str | pathlib.Path, transform: points_alignment.SimilarityTransform
) -> None:
    np.savez_compressed(
        file, R=transform.R.numpy(), T=transform.T.numpy(), s=transform.s.numpy()
    )


def load(file: str | pathlib.Path) -> points_alignment.SimilarityTransform:
    data: npyio.NpzFile = np.load(file)
    return points_alignment.SimilarityTransform(
        R=torch.tensor(data["R"]), T=torch.tensor(data["T"]), s=torch.tensor(data["s"])
    )


def transform_points(
    transform: points_alignment.SimilarityTransform, points: torch.Tensor
) -> torch.Tensor:
    points = transform.s[:, None, None] * points
    points = torch.bmm(points, transform.R) + transform.T[:, None, :]
    return points
