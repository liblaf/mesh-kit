from typing import Any

import pyvista as pv

import mkit
import mkit.typing.numpy as tn


def mask_points(mesh: Any, mask: tn.BNLike | None = None) -> pv.PolyData:
    mesh: pv.PolyData = mkit.io.pyvista.as_poly_data(mesh)
    if mask is None:
        return mesh
    mask: tn.BN = mkit.math.numpy.as_bool(mask)
    ug: pv.UnstructuredGrid = mesh.extract_points(mask)
    mesh = ug.extract_surface()
    return mesh
