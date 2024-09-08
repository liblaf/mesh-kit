from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pyvista as pv

import mkit
from mkit.ops.transfer import surface_to_surface
from mkit.typing import AnyTetMesh, AnyTriMesh


def surface_to_tetmesh(
    _source: AnyTriMesh,
    _target: AnyTetMesh,
    point_data_names: Sequence[str] | None = None,
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
        data: npt.NDArray = source.point_data[k]
        target.point_data[k] = np.zeros((target.n_points, *data.shape[1:]), data.dtype)
        target.point_data[k][original_point_ids] = surface.point_data[k]
    return target
