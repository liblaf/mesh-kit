from typing import Any

import pyvista as pv

import mkit
import mkit.typing.numpy as nt


def mask_points(
    mesh: Any, mask: nt.BNLike | None = None, *, progress_bar: bool = False
) -> pv.PolyData:
    mesh: pv.PolyData = mkit.io.pyvista.as_poly_data(mesh)
    if mask is None:
        return mesh
    mask: nt.BN = mkit.math.numpy.as_bool(mask)
    unstructured: pv.UnstructuredGrid = mesh.extract_points(
        mask, progress_bar=progress_bar
    )
    mesh = unstructured.extract_surface(progress_bar=progress_bar)
    return mesh
