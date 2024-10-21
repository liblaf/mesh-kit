from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mkit.io as mi
import mkit.math as mm
import mkit.typing.numpy as tn

if TYPE_CHECKING:
    import pyvista as pv


def transform(
    mesh: Any,
    transformation: tn.F44Like | None = None,
    *,
    transform_all_input_vectors: bool = False,
) -> pv.PolyData:
    mesh: pv.PolyData = mi.pyvista.as_poly_data(mesh)
    if transformation is None:
        return mesh
    mesh = mesh.transform(
        mm.as_numpy(transformation),
        transform_all_input_vectors=transform_all_input_vectors,
        inplace=False,
    )  # pyright: ignore [reportAssignmentType]
    return mesh
