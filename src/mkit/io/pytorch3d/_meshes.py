from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from pytorch3d.structures import Meshes

import mkit.io._register as r
from mkit.io._typing import ClassName as C  # noqa: N814

if TYPE_CHECKING:
    import pyvista as pv


def as_meshes(mesh: Any) -> Meshes:
    return r.convert(mesh, Meshes)


@r.register(C.PYVISTA_POLY_DATA, C.PYTORCH3D)
def pyvista_polydata_to_pytorch3d(
    mesh: pv.PolyData, *, progress_bar: bool = False
) -> Meshes:
    mesh = mesh.triangulate(progress_bar=progress_bar)
    points: torch.Tensor = torch.as_tensor(mesh.points).reshape(1, -1, 3)
    faces: torch.Tensor = torch.as_tensor(mesh.regular_faces).unsqueeze(0)
    return Meshes(points, faces)
