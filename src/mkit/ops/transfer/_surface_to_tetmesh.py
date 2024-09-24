from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pyvista as pv

import mkit
from mkit.typing import AttributesLike


def surface_to_tetmesh(
    source: Any,
    target: Any,
    point_data: AttributesLike | None = None,
    point_data_names: Iterable[str] | None = None,
    cell_data: AttributesLike | None = None,
    cell_data_names: Iterable[str] | None = None,
    *,
    distance_threshold: float = 0.1,
    method: Literal["auto", "barycentric", "nearest"] = "auto",
    transfer: Literal["point-to-point", "cell-to-cell"] = "point-to-point",
) -> pv.UnstructuredGrid:
    source: pv.PolyData = mkit.io.pyvista.as_poly_data(source)
    target: pv.UnstructuredGrid = mkit.io.pyvista.as_unstructured_grid(target)

    surface: pv.PolyData = target.extract_surface(progress_bar=True)
    original_point_id: npt.NDArray[np.integer] = surface.point_data[
        "vtkOriginalPointIds"
    ]
    surface = mkit.ops.transfer.surface_to_surface(
        source,
        surface,
        point_data=point_data,
        point_data_names=point_data_names,
        cell_data=cell_data,
        cell_data_names=cell_data_names,
        distance_threshold=distance_threshold,
        method=method,
        transfer=transfer,
    )
    for k, v in surface.point_data.items():
        target.point_data[k] = np.zeros((target.n_points, *v.shape[1:]), v.dtype)
        target.point_data[k][original_point_id] = v
    if surface.cell_data:
        raise NotImplementedError
    return target
