from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mkit.io as mi
import mkit.typing.numpy as tn

if TYPE_CHECKING:
    import pyvista as pv


def point_area(mesh: Any) -> tn.FN:
    mesh_pv: pv.PolyData = mi.pyvista.as_poly_data(mesh)
    mesh_pv = mesh_pv.compute_cell_sizes(
        length=False, area=True, volume=False, vertex_count=False
    )  # pyright: ignore [reportAssignmentType]
    mesh_pv = mesh_pv.cell_data_to_point_data(pass_cell_data=False)  # pyright: ignore [reportAssignmentType]
    return mesh_pv.point_data["Area"]
