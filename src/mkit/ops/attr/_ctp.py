from typing import TYPE_CHECKING, Any

import numpy as np

import mkit
from mkit.typing import AttributeArray, AttributesLike

if TYPE_CHECKING:
    import pyvista as pv


def cell_data_to_point_data(
    mesh: Any, cell_data: AttributesLike | None = None
) -> dict[str, AttributeArray]:
    mesh: pv.PolyData = mkit.io.pyvista.as_poly_data(mesh).copy()
    if cell_data is not None:
        mesh.cell_data.clear()
        mesh.cell_data.update({k: np.asarray(v) for k, v in cell_data.items()})
    mesh = mesh.cell_data_to_point_data()
    return {k: np.asarray(v) for k, v in mesh.point_data.items()}
