import numpy as np
import pyvista as pv
import trimesh
from numpy import typing as npt

from mesh_kit.typing import check_type as _check_type


def as_trimesh(obj) -> trimesh.Trimesh:
    match obj:
        case pv.PolyData():
            mesh: pv.PolyData = _check_type(obj.triangulate(), pv.PolyData)
            return trimesh.Trimesh(
                vertices=mesh.points,
                faces=_check_type(mesh.regular_faces, npt.NDArray[np.int64]),
            )
        case _:
            raise NotImplementedError


def copy(
    mesh: trimesh.Trimesh, *, cache: bool = False, attrs: bool = True
) -> trimesh.Trimesh:
    copy: trimesh.Trimesh = mesh.copy(include_cache=cache)
    if attrs:
        copy.vertex_attributes = mesh.vertex_attributes.copy()
        copy.face_attributes = mesh.face_attributes.copy()
    return copy
