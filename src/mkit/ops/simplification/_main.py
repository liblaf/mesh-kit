from __future__ import annotations

from typing import TYPE_CHECKING, Any

import fast_simplification

import mkit.io as mi

if TYPE_CHECKING:
    import pyvista as pv


def simplify(
    mesh: Any,
    *,
    agg: int = 7,
    target_count: int | None = None,
    target_density: float | None = 1e4,
    target_reduction: float | None = None,
    verbose: bool = False,
) -> pv.PolyData:
    mesh: pv.PolyData = mi.pyvista.as_poly_data(mesh)
    mesh.triangulate(inplace=True)
    if target_count is None and target_density is not None and target_reduction is None:
        target_count = int(target_density * (mesh.area / mesh.length**2))
    if target_count is not None and target_count >= mesh.n_faces_strict:
        return mesh
    mesh = fast_simplification.simplify_mesh(
        mesh,
        target_reduction=target_reduction,
        target_count=target_count,
        agg=agg,
        verbose=verbose,
    )
    return mesh
