from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import mkit.io as mi
import mkit.math as mm
import mkit.ops as mo
import mkit.typing.numpy as tn

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pyvista as pv


def mask_by_group_ids(
    mesh: Any,
    selection: tn.INLike,
    *,
    group_ids: tn.INLike | None = None,
    invert: bool = False,
) -> tn.BN:
    if group_ids is None:
        mesh_pv: pv.PolyData = mi.pyvista.as_poly_data(mesh)
        group_ids = mesh_pv.cell_data["GroupIds"]
    selection: tn.IN = mm.as_numpy(selection)
    group_ids: tn.IN = mm.as_numpy(group_ids)
    return np.isin(group_ids, selection, invert=invert)


def mask_by_group_names(
    mesh: Any,
    selection: Iterable[str],
    *,
    group_ids: tn.INLike | None = None,
    group_names: Sequence[str] | None = None,
    invert: bool = False,
) -> tn.BN:
    if group_names is None:
        mesh_pv: pv.PolyData = mi.pyvista.as_poly_data(mesh)
        group_names = mesh_pv.field_data["GroupNames"]  # pyright: ignore [reportAssignmentType]
    group_names: Sequence[str]
    names_to_ids: dict[str, int] = {name: i for i, name in enumerate(group_names)}
    selection_ids: list[int] = [names_to_ids[name] for name in selection]
    return mask_by_group_ids(mesh, selection_ids, group_ids=group_ids, invert=invert)


def select_by_group_ids(
    mesh: Any,
    selection: tn.INLike,
    *,
    group_ids: tn.INLike | None = None,
    invert: bool = False,
) -> pv.PolyData:
    mask: tn.BN = mask_by_group_ids(mesh, selection, group_ids=group_ids, invert=invert)
    return mo.select_cells(mesh, mask)


def select_by_group_names(
    mesh: Any,
    selection: Sequence[str],
    *,
    group_ids: tn.INLike | None = None,
    group_names: Sequence[str] | None = None,
    invert: bool = False,
) -> pv.PolyData:
    mask: tn.BN = mask_by_group_names(
        mesh, selection, group_ids=group_ids, group_names=group_names, invert=invert
    )
    return mo.select_cells(mesh, mask)
