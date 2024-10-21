from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

import mkit.io as mi
import mkit.math as mm
import mkit.typing.numpy as tn

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyvista as pv

_T = TypeVar("_T")


def select_by_group_ids(
    mesh: _T,
    selection: tn.INLike,
    *,
    group_ids: tn.INLike | None = None,
    invert: bool = False,
) -> _T:
    mesh_pv: pv.PolyData = mi.pyvista.as_poly_data(mesh)
    group_ids: tn.IN
    if group_ids is None:
        group_ids = mesh_pv.cell_data["GroupIds"]
    else:
        group_ids = mm.as_numpy(group_ids)
    unstructured_grid: pv.UnstructuredGrid = mesh_pv.extract_cells(
        np.isin(group_ids, selection), invert=invert
    )
    result: pv.PolyData = unstructured_grid.extract_surface()
    return mi.convert(result, type(mesh))


def select_by_group_names(
    mesh: _T,
    selection: Sequence[str],
    *,
    group_ids: tn.INLike | None = None,
    group_names: Sequence[str] | None = None,
    invert: bool = False,
) -> _T:
    if group_names is None:
        mesh_pv: pv.PolyData = mi.pyvista.as_poly_data(mesh)
        group_names = mesh_pv.field_data["GroupNames"]
    group_names: list[str] = list(group_names)
    selection_ids: list[int] = [group_names.index(name) for name in selection]
    return select_by_group_ids(mesh, selection_ids, group_ids=group_ids, invert=invert)
