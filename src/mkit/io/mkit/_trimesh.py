from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mkit
from mkit.io._register import REGISTRY
from mkit.io._typing import ClassName as C  # noqa: N814

if TYPE_CHECKING:
    import pyvista as pv
    import trimesh as tm


def as_trimesh(mesh: Any) -> mkit.TriMesh:
    return REGISTRY.convert(mesh, mkit.TriMesh)


@REGISTRY.register(C.TRIMESH, C.MKIT_TRIMESH)
def trimesh_to_trimesh(mesh: tm.Trimesh) -> mkit.TriMesh:
    return mkit.TriMesh(mesh.vertices, mesh.faces)


def polydata_to_trimesh(mesh: pv.PolyData) -> mkit.TriMesh:
    mesh = mesh.triangulate()
    return mkit.TriMesh(mesh.points, mesh.faces)
