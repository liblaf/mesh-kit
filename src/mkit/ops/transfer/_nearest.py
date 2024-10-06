import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
from scipy.spatial import KDTree

import mkit
import mkit.typing.numpy as nt
from mkit.ops.transfer._abc import C2CMethod, P2PMethod
from mkit.typing import AttributeArray, AttributesLike

if TYPE_CHECKING:
    import pyvista as pv


@dataclasses.dataclass(kw_only=True)
class C2CNearest(C2CMethod):
    distance_threshold: float = 0.1
    fill_value: nt.ArrayLike = np.nan

    def __call__(
        self, source: Any, target: Any, data: AttributesLike | None = None
    ) -> dict[str, AttributeArray]:
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
        valid: nt.BN = dist < self.distance_threshold * source.length
        if not np.all(valid):
            logger.warning(
                "Some cells are not within the distance threshold: {}",
                self.distance_threshold,
            )
        target_cell_data: dict[str, nt.Shaped[np.ndarray, "V ..."]] = {}
        for k, v in data.items():
            data: nt.Shaped[np.ndarray, "V ..."] = np.asarray(v)[idx]
            target_cell_data[k] = data
            if not valid.all():
                target_cell_data[k][~valid] = self.fill_value
        return target_cell_data


@dataclasses.dataclass(kw_only=True)
class P2PNearest(P2PMethod):
    distance_threshold: float = 0.1
    fill_value: nt.ArrayLike = np.nan

    def __call__(
        self, source: Any, target: Any, data: AttributesLike | None = None
    ) -> dict[str, AttributeArray]:
        if not data:
            return {}
        raise NotImplementedError
