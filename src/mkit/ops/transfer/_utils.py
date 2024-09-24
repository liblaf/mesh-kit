from collections.abc import Iterable

import numpy as np
import pyvista as pv

import mkit.typing.numpy as nt
from mkit.typing import AttributesLike


def get_point_data(
    mesh: pv.PolyData, data: AttributesLike | None, data_names: Iterable[str] | None
) -> dict[str, nt.FN]:
    if data is not None:
        return {k: np.asarray(v) for k, v in data.items()}
    if data_names is not None:
        return {k: np.asarray(mesh.point_data[k]) for k in data_names}
    return dict(mesh.point_data)


def get_cell_data(
    mesh: pv.PolyData, data: AttributesLike | None, data_names: Iterable[str] | None
) -> dict[str, nt.FN]:
    if data is not None:
        return {k: np.asarray(v) for k, v in data.items()}
    if data_names is not None:
        return {k: np.asarray(mesh.cell_data[k]) for k in data_names}
    return dict(mesh.cell_data)
