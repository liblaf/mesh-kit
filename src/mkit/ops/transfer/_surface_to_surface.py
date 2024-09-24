from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import pyvista as pv

import mkit
import mkit.ops.transfer._auto as auto
import mkit.ops.transfer._barycentric as barycentric
import mkit.ops.transfer._nearest as nearest
import mkit.ops.transfer._utils as u
import mkit.typing.numpy as nt
from mkit.ops.transfer._abc import TransferFn
from mkit.typing import AttributesLike

if TYPE_CHECKING:
    import numpy as np
    from jaxtyping import Shaped

_METHODS: dict[tuple[str, str], TransferFn] = {
    ("auto", "point-to-point"): auto.point_to_point,
    ("auto", "cell-to-cell"): auto.cell_to_cell,
    ("barycentric", "point-to-point"): barycentric.point_to_point,
    ("barycentric", "cell-to-cell"): barycentric.cell_to_cell,
    ("nearest", "point-to-point"): nearest.point_to_point,
    ("nearest", "cell-to-cell"): nearest.cell_to_cell,
}


def surface_to_surface(
    source: Any,
    target: Any,
    point_data: AttributesLike | None = None,
    point_data_names: Iterable[str] | None = None,
    cell_data: AttributesLike | None = None,
    cell_data_names: Iterable[str] | None = None,
    *,
    distance_threshold: float = 0.1,
    method: Literal["auto", "barycentric", "nearest"] = "auto",
    transfer: Literal["point-to-point", "cell-to-cell"] = "point-to-point",
) -> pv.PolyData:
    source: pv.PolyData = mkit.io.pyvista.as_poly_data(source)
    target: pv.PolyData = mkit.io.pyvista.as_poly_data(target)
    point_data: dict[str, nt.FN] = u.get_point_data(
        source, point_data, point_data_names
    )
    cell_data: dict[str, nt.FN] = u.get_cell_data(source, cell_data, cell_data_names)
    if (method, transfer) not in _METHODS:
        msg: str = f"Unsupported transfer: {method} {transfer}"
        raise ValueError(msg)
    fn: TransferFn = _METHODS[(method, transfer)]
    data: dict[str, Shaped[np.ndarray, "N ..."]] = fn(
        source, target, point_data, distance_threshold=distance_threshold
    )
    if transfer.endswith("to-point"):
        target.point_data.update(data)
    elif transfer.endswith("to-cell"):
        target.cell_data.update(data)
    return target
