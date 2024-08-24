from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pyvista as pv
import trimesh

import mkit.io


def surface_to_tetmesh(
    _source: Any, _target: Any, point_data_names: Sequence[str] | None = None
) -> pv.UnstructuredGrid:
    source: pv.PolyData = mkit.io.as_polydata(_source)
    target: pv.UnstructuredGrid = mkit.io.as_unstructured_grid(_target)
    if point_data_names is None:
        point_data_names = source.point_data.keys()
    surface: pv.PolyData = target.extract_surface(progress_bar=True)
    original_point_ids: npt.NDArray[np.integer] = surface.point_data[
        "vtkOriginalPointIds"
    ]
    surface = surface_to_surface(_source, surface, point_data_names=point_data_names)
    for k in point_data_names:
        data: npt.NDArray[...] = source.point_data[k]
        target.point_data[k] = np.zeros((target.n_points, *data.shape[1:]), data.dtype)
        target.point_data[k][original_point_ids] = surface.point_data[k]
    return target


def surface_to_surface(
    _source: Any, _target: Any, point_data_names: Sequence[str] | None = None
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
        data: npt.NDArray[...] = source.point_data[k][faces]  # (V, 3, ...)
        data_interp: npt.NDArray[...] = np.einsum("ij,ij...->i...", barycentric, data)
        if np.isdtype(data.dtype, "bool"):
            target[k] = data_interp > 0.5
        else:
            target[k] = data_interp.astype(data.dtype)
    return target
