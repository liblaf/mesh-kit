from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import fast_simplification

import mkit.io as mi

if TYPE_CHECKING:
    import pyvista as pv

_T = TypeVar("_T")


def simplify(
    mesh: _T,
    *,
    agg: int = 7,
    target_count: int | None = None,
    target_density: float | None = 1e4,
    target_reduction: float | None = None,
    verbose: bool = False,
) -> _T:
    mesh_pv: pv.PolyData = mi.pyvista.as_poly_data(mesh)
    mesh_pv.triangulate(inplace=True)
    if target_count is None and target_density is not None and target_reduction is None:
        target_count = int(target_density * (mesh_pv.area / mesh_pv.length**2))
    if target_count is not None and target_count >= mesh_pv.n_faces_strict:
        return mesh
    mesh_pv = fast_simplification.simplify_mesh(
        mesh_pv,
        target_reduction=target_reduction,
        target_count=target_count,
        agg=agg,
        verbose=verbose,
    )
    return mi.convert(mesh_pv, type(mesh))
