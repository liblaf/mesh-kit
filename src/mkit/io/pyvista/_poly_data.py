from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv

import mkit
import mkit.typing.numpy as nt
from mkit.io._register import REGISTRY
from mkit.io._typing import ClassName as C  # noqa: N814

if TYPE_CHECKING:
    import trimesh as tm

    from mkit.typing import AnySurfaceMesh, StrPath


def as_poly_data(mesh: AnySurfaceMesh) -> pv.PolyData:
    return REGISTRY.convert(mesh, pv.PolyData)


def is_point_cloud(mesh: pv.PolyData) -> bool:
    """Determine if a given PyVista PolyData object represents a point cloud.

    A point cloud is characterized by having the number of points plus the number of lines
    equal to the number of cells.

    Args:
        mesh: The PyVista PolyData object to check.

    Returns:
        True if the PolyData object is a point cloud, False otherwise.

    Reference:
        1. <https://github.com/pyvista/pyvista/blob/556d86725f27c43e30445c877961910d0fff0893/pyvista/core/filters/poly_data.py#L1730-L1734>
    """
    return (mesh.n_points + mesh.n_lines) == mesh.n_cells


def load_poly_data(fpath: StrPath) -> pv.PolyData:
    fpath: Path = Path(fpath)
    mesh: pv.PolyData
    match fpath.suffix:
        case ".obj":
            mesh = mkit.io.pyvista.load_obj(fpath)
        case _:
            mesh = pv.read(fpath)
    mesh = mesh.clean()
    return mesh


REGISTRY.register(C.MESHIO, C.PYVISTA_POLY_DATA)(pv.wrap)


@REGISTRY.register(C.ARRAY_LIKE, C.PYVISTA_POLY_DATA, priority=-10)
def array_to_poly_data(points: nt.FN3Like) -> pv.PolyData:
    return pv.wrap(np.asarray(points))  # pyright: ignore [reportReturnType]


@REGISTRY.register(C.TRIMESH, C.PYVISTA_POLY_DATA)
def trimesh_to_poly_data(mesh: tm.Trimesh) -> pv.PolyData:
    try:
        wrapped: pv.PolyData = pv.wrap(mesh)
    except NotImplementedError:
        return pv.make_tri_mesh(mesh.vertices, mesh.faces)
    else:
        return wrapped


@REGISTRY.register(C.OPEN3D_POINT_CLOUD, C.PYVISTA_POLY_DATA)
def open3d_point_cloud_to_poly_data(mesh: Any) -> pv.PolyData:
    pcd: pv.PolyData = pv.wrap(mesh.points)
    return pcd
