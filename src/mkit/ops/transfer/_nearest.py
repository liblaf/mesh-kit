from typing import TYPE_CHECKING, Any

import numpy as np
from jaxtyping import Shaped
from loguru import logger
from scipy.spatial import KDTree

import mkit
import mkit.typing.numpy as nt
from mkit.typing import AttributesLike

if TYPE_CHECKING:
    import pyvista as pv


def point_to_point(
    source: Any,
    target: Any,
    data: AttributesLike | None = None,
    *,
    distance_threshold: float = 0.1,
) -> dict[str, Shaped[np.ndarray, "V ..."]]:
    if not data:
        return {}
    raise NotImplementedError


def cell_to_cell(
    source: Any,
    target: Any,
    data: AttributesLike | None = None,
    *,
    distance_threshold: float = 0.1,
) -> dict[str, Shaped[np.ndarray, "V ..."]]:
    if not data:
        return {}
    source: pv.PolyData = mkit.io.pyvista.as_poly_data(source)
    target: pv.PolyData = mkit.io.pyvista.as_poly_data(target)
    source_centers: pv.PolyData = source.cell_centers()
    target_centers: pv.PolyData = target.cell_centers()
    tree: KDTree = KDTree(source_centers.points)
    dist: nt.FN
    idx: nt.IN
    dist, idx = tree.query(target_centers.points)
    valid: nt.BN = dist < distance_threshold * source.length
    if not np.all(valid):
        logger.warning(
            "Some cells are not within the distance threshold: {}", distance_threshold
        )
    target_cell_data: dict[str, Shaped[np.ndarray, "V ..."]] = {}
    for k, v in data.items():
        data: Shaped[np.ndarray, "V ..."] = np.asarray(v)[idx]
        target_cell_data[k] = data
        target_cell_data[k][~valid] = np.nan
    return target_cell_data
