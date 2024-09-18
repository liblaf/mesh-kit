from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np
import pyvista as pv

import mkit
import mkit.io._register as r
import mkit.typing.numpy as nt
from mkit.io._typing import ClassName as C  # noqa: N814

if TYPE_CHECKING:
    import trimesh as tm

    from mkit.typing import AnySurfaceMesh, StrPath


def as_poly_data(mesh: AnySurfaceMesh) -> pv.PolyData:
    return r.convert(mesh, pv.PolyData)


def is_poly_data(mesh: Any) -> TypeGuard[pv.PolyData]:
    return isinstance(mesh, pv.PolyData)


def is_point_cloud(mesh: pv.PolyData) -> bool:
    """Determine if a given PyVista PolyData object represents a point cloud.

    A point cloud is characterized by having the number of points plus the number of lines
    equal to the number of cells.

    Args:
        mesh: The PyVista PolyData object to check.

    Returns:
        True if the PolyData object is a point cloud, False otherwise.

    Reference:
        1. <https://github.com/pyvista/pyvista/blob/main/pyvista/core/filters/poly_data.py#L1724-L1728>
    """
    return (mesh.n_points + mesh.n_lines) == mesh.n_cells


def read_poly_data(fpath: StrPath) -> pv.PolyData:
    fpath: Path = Path(fpath)
    if fpath.suffix == ".obj":
        return read_obj(fpath)
    mesh: pv.PolyData = pv.read(fpath)
    return mesh


def read_obj(fpath: StrPath) -> pv.PolyData:
    fpath: Path = Path(fpath)
    group_dup: list[str] = []
    for line in mkit.utils.strip_comments(fpath.read_text()):
        if line.startswith("g"):
            words: list[str] = line.split()
            if len(words) >= 2:
                group_dup.append(words[1])
            else:
                group_dup.append(str(len(group_dup)))
    mesh: pv.PolyData
    if not group_dup:
        mesh = pv.read(fpath)
        return mesh
    group_uniq: list[str] = list(dict.fromkeys(group_dup))
    dup_to_uniq: dict[str, int] = {name: i for i, name in enumerate(group_uniq)}
    mesh = pv.read(fpath)
    group_id: nt.IN = mkit.math.numpy.cast(mesh.cell_data["GroupIds"], int)
    group_id = np.asarray([dup_to_uniq[group_dup[i]] for i in group_id])
    mesh.cell_data["GroupIds"] = group_id
    mesh.field_data["GroupNames"] = group_uniq
    return mesh


r.register(C.MESHIO, C.PYVISTA_POLY_DATA)(pv.wrap)


@r.register(C.TRIMESH, C.PYVISTA_POLY_DATA)
def trimesh_to_poly_data(mesh: tm.Trimesh) -> pv.PolyData:
    try:
        wrapped: pv.PolyData = pv.wrap(mesh)
    except NotImplementedError:
        return pv.make_tri_mesh(mesh.vertices, mesh.faces)
    else:
        return wrapped


@r.register(C.OPEN3D_POINT_CLOUD, C.PYVISTA_POLY_DATA)
def open3d_point_cloud_to_poly_data(mesh: Any) -> pv.PolyData:
    pcd: pv.PolyData = pv.wrap(mesh.points)
    return pcd
