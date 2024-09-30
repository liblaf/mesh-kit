from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from pytorch3d.structures import Meshes

from mkit.io._register import REGISTRY
from mkit.io._typing import ClassName as C  # noqa: N814

if TYPE_CHECKING:
    import pyvista as pv


def as_meshes(mesh: Any) -> Meshes:
    return REGISTRY.convert(mesh, Meshes)


@REGISTRY.register(C.PYVISTA_POLY_DATA, C.PYTORCH3D_MESHES)
def pyvista_poly_data_to_pytorch3d(
    mesh: pv.PolyData, *, progress_bar: bool = False
) -> Meshes:
    mesh = mesh.triangulate(progress_bar=progress_bar)
    points: torch.Tensor = torch.as_tensor(mesh.points).reshape(1, mesh.n_points, 3)
    faces: torch.Tensor = torch.as_tensor(mesh.regular_faces).reshape(
        1, mesh.n_faces_strict, 3
    )
    return Meshes(points, faces)
