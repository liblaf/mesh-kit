from __future__ import annotations

from typing import TYPE_CHECKING

import trimesh as tm

import mkit.io._register as r
from mkit.io._typing import ClassName as C  # noqa: N814

if TYPE_CHECKING:
    import meshio
    import pyvista as pv

    from mkit.typing import AnySurfaceMesh


def as_trimesh(mesh: AnySurfaceMesh) -> tm.Trimesh:
    return r.convert(mesh, tm.Trimesh)


@r.register(C.MESHIO, C.TRIMESH)
def meshio_to_trimesh(mesh: meshio.Mesh) -> tm.Trimesh:
    return tm.Trimesh(mesh.points, mesh.get_cells_type("triangle"))


@r.register(C.PYVISTA_POLY_DATA, C.TRIMESH)
def polydata_to_trimesh(mesh: pv.PolyData, *, progress_bar: bool = True) -> tm.Trimesh:
    mesh = mesh.triangulate(progress_bar=progress_bar)
    return tm.Trimesh(mesh.points, mesh.regular_faces)
