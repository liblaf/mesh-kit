from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

import mkit
import mkit.typing.numpy as tn

if TYPE_CHECKING:
    import pyvista as pv


def names_to_ids(
    names: Sequence[str],
    *,
    mesh: Any | None = None,
    group_names: Sequence[str] | None = None,
) -> list[int]:
    if group_names is None:
        mesh: pv.PolyData = mkit.io.pyvista.as_poly_data(mesh)
        group_names = mesh.field_data["GroupNames"]
    group_names: list[str] = list(group_names)
    return [group_names.index(n) for n in names]


def select_by_group_ids(
    ids: Sequence[int], *, mesh: Any | None = None, group_ids: tn.INLike | None = None
) -> tn.BN:
    if group_ids is None:
        mesh: pv.PolyData = mkit.io.pyvista.as_poly_data(mesh)
        group_ids = mesh.cell_data["GroupIds"]
    return np.isin(group_ids, ids)


def select_by_group_names(
    names: Sequence[str],
    *,
    mesh: Any | None = None,
    group_ids: tn.INLike | None = None,
    group_names: tn.Shaped[tn.ArrayLike, " N"] | None = None,
) -> tn.BN:
    return select_by_group_ids(
        names_to_ids(names, mesh=mesh, group_names=group_names),
        mesh=mesh,
        group_ids=group_ids,
    )
