from __future__ import annotations

from typing import TYPE_CHECKING, Any

import attrs
import numpy as np
import trimesh as tm

import mkit.io as mi
from mkit.ops.register import rigid

if TYPE_CHECKING:
    import mkit.typing.numpy as tn


@attrs.define
class RigidICP(rigid.RigidRegistrationBase):
    distance_threshold: float = 0.1
    loss_threshold: float = 1e-6
    max_iters: int = 100
    normal_threshold: float = 0.8

    def _register(
        self,
        source: Any,
        target: Any,
        *,
        source_weights: tn.FNLike | None = None,
        target_weights: tn.FNLike | None = None,
    ) -> rigid.RigidRegistrationResult:
        source: tm.Trimesh = mi.trimesh.as_trimesh(source)
        target: tm.Trimesh = mi.trimesh.as_trimesh(target)
        result: rigid.RigidRegistrationResult = rigid.RigidRegistrationResult(
            loss=np.nan, transformation=np.eye(4)
        )
        for it in range(self.max_iters):
            closest: tn.FN3
            distance: tn.FN
            triangle_id: tn.FN
            closest, distance, triangle_id = target.nearest.on_surface(source.vertices)
            valid_mask: tn.BN = (distance < self.distance_threshold) & (
                np.vecdot(source.vertex_normals, target.face_normals[triangle_id])
                > self.normal_threshold
            )
            matrix: tn.F44
            cost: float
            if source_weights is None:
                source_weights = np.ones(len(source.vertices))
            matrix, _transformed, cost = tm.registration.procrustes(
                source.vertices[valid_mask],
                closest[valid_mask],
                weights=None
                if self.source_weights is None
                else self.source_weights[valid_mask],
                reflection=self.reflection,
                translation=self.translation,
                scale=self.scale,
                return_cost=True,
            )
            ic(cost)
            result.transformation = matrix @ result.transformation
            if result.loss - cost < self.loss_threshold:
                break
            result.loss = cost
            source = source.apply_transform(matrix)
        return result
