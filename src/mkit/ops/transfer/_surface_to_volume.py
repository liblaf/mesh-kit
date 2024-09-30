from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

import mkit
from mkit.ops.transfer import C2CMethod, P2PMethod
from mkit.typing import AttributesLike
from mkit.typing._geometry import AttributeArray

if TYPE_CHECKING:
    import pyvista as pv


def surface_to_volume(
    source: Any,
    target: Any,
    data: AttributesLike | None = None,
    *,
    method: C2CMethod | P2PMethod,
) -> dict[str, AttributeArray]:
    if isinstance(method, C2CMethod):
        raise NotImplementedError
    source: pv.PolyData = mkit.io.pyvista.as_poly_data(source)
    target: pv.UnstructuredGrid = mkit.io.pyvista.as_unstructured_grid(target)
    surface: pv.PolyData = target.extract_surface()
    original_point_ids: npt.NDArray[np.integer] = surface.point_data[
        "vtkOriginalPointIds"
    ]
    surface_data: dict[str, AttributeArray] = mkit.ops.transfer.surface_to_surface(
        source, surface, data, method=method
    )
    volume_data: dict[str, AttributeArray] = {}
    for k, v in surface_data.items():
        volume_data[k] = np.zeros((target.n_points, *v.shape[1:]), v.dtype)
        volume_data[k][original_point_ids] = v
    return volume_data
