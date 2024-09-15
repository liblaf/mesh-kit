from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pyvista as pv

import mkit
import mkit.typing as t


def surface_to_tetmesh(
    source: t.AnyTriMesh,
    target: t.AnyTetMesh,
    point_data: Mapping[str, npt.ArrayLike] | pv.DataSetAttributes | None = None,
    point_data_names: Sequence[str] | None = None,
) -> pv.UnstructuredGrid:
    source: pv.PolyData = mkit.io.pyvista.as_poly_data(source)
    target: pv.UnstructuredGrid = mkit.io.pyvista.as_unstructured_grid(target)
    if point_data is None:
        if point_data_names is not None:
            point_data = {k: source.point_data[k] for k in point_data_names}
        else:
            point_data = source.point_data
    surface: pv.PolyData = target.extract_surface(progress_bar=True)
    original_point_id: npt.NDArray[np.integer] = surface.point_data[
        "vtkOriginalPointIds"
    ]
    surface = mkit.ops.transfer.surface_to_surface(
        source, surface, point_data=point_data, point_data_names=point_data_names
    )
    for k in point_data:
        data: npt.NDArray = surface.point_data[k]
        target.point_data[k] = np.zeros((target.n_points, *data.shape[1:]), data.dtype)
        target.point_data[k][original_point_id] = surface.point_data[k]
    return target
