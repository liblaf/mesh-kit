from typing import Any

import pyvista as pv

from mkit.io.typing import UnsupportedMeshError, is_meshio, is_polydata, is_trimesh


def as_polydata(mesh: Any) -> pv.PolyData:
    if is_polydata(mesh):
        return mesh
    if is_trimesh(mesh):
        return pv.wrap(mesh)  # pyright: ignore [reportReturnType]
    if is_meshio(mesh):
        return pv.wrap(mesh)  # pyright: ignore [reportReturnType]
    raise UnsupportedMeshError(mesh)
