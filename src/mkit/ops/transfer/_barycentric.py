import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import trimesh as tm
from loguru import logger

import mkit
import mkit.typing.numpy as nt
from mkit.ops.transfer._abc import C2CMethod, P2PMethod
from mkit.typing import AttributeArray, AttributesLike

if TYPE_CHECKING:
    from jaxtyping import Shaped


@dataclasses.dataclass(kw_only=True)
class C2CBarycentric(C2CMethod):
    distance_threshold: float = 0.1

    def __call__(
        self, source: Any, target: Any, data: AttributesLike | None = None
    ) -> dict[str, AttributeArray]:
        raise NotImplementedError


@dataclasses.dataclass(kw_only=True)
class P2PBarycentric(P2PMethod):
    distance_threshold: float = 0.1

    def __call__(
        self, source: Any, target: Any, data: AttributesLike | None = None
    ) -> dict[str, AttributeArray]:
        if not data:
            return {}
        source: tm.Trimesh = mkit.io.trimesh.as_trimesh(source)
        target: tm.Trimesh = mkit.io.trimesh.as_trimesh(target)
        closest: nt.FN3
        dist: nt.FN
        triangle_id: nt.IN
        closest, dist, triangle_id = source.nearest.on_surface(target.vertices)
        faces: nt.IN3 = source.faces[triangle_id]
        barycentric: nt.FN3 = tm.triangles.points_to_barycentric(
            source.vertices[faces], closest
        )
        target_point_data: dict[str, AttributeArray] = {}
        valid: nt.BN = dist < self.distance_threshold * source.scale
        if not np.all(valid):
            logger.warning(
                "Some points are not within the distance threshold: {}",
                self.distance_threshold,
            )
        for k, v in data.items():
            data: Shaped[np.ndarray, "V 3 ..."] = np.asarray(v)[faces]  # (V, 3, ...)
            data_interp: Shaped[np.ndarray, "V 3 ..."] = np.einsum(
                "ij,ij...->i...", barycentric, data
            )
            target_point_data[k] = mkit.math.numpy.cast(data_interp, data.dtype)
            target_point_data[k][~valid] = np.nan
        return target_point_data
