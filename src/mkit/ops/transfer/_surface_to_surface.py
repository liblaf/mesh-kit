from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pyvista as pv
import trimesh

import mkit
from mkit.io import AnyTriMesh


def surface_to_surface(
    _source: AnyTriMesh,
    _target: AnyTriMesh,
    point_data_names: Sequence[str] | None = None,
) -> pv.PolyData:
    source: pv.PolyData = mkit.io.as_polydata(_source)
    target: pv.PolyData = mkit.io.as_polydata(_target)
    if point_data_names is None:
        point_data_names = source.point_data.keys()
    source_tr: trimesh.Trimesh = mkit.io.as_trimesh(_source)
    target_tr: trimesh.Trimesh = mkit.io.as_trimesh(_target)
    closest: npt.NDArray[np.floating]  # (V, 3)
    distance: npt.NDArray[np.floating]  # (V,)
    triangle_id: npt.NDArray[np.integer]  # (V,)
    closest, distance, triangle_id = source_tr.nearest.on_surface(target_tr.vertices)
    faces: npt.NDArray[np.integer] = source_tr.faces[triangle_id]  # (V, 3)
    barycentric: npt.NDArray[np.floating] = trimesh.triangles.points_to_barycentric(
        source_tr.vertices[faces], closest
    )  # (V, 3)
    for k in point_data_names:
        data: npt.NDArray = source.point_data[k][faces]  # (V, 3, ...)
        data_interp: npt.NDArray = np.einsum("ij,ij...->i...", barycentric, data)
        target[k] = mkit.math.numpy.cast(data_interp, data.dtype)
    return target
