import dataclasses
from typing import Any

import trimesh as tm
from loguru import logger

import mkit
import mkit.typing.numpy as tn
from mkit.ops.registration import rigid


@dataclasses.dataclass(kw_only=True)
class TrimeshICP(rigid.RigidRegistrationMethod):
    max_iterations: int = 100
    reflection: bool = False
    scale: bool = True
    threshold: float = 1e-6
    translation: bool = True

    def __call__(
        self,
        source: Any,
        target: Any,
        source_weight: tn.FNLike | None = None,
        target_weight: tn.FNLike | None = None,
    ) -> rigid.RigidRegistrationResult:
        source: tm.Trimesh = _preprocess(source, source_weight)
        target: tm.Trimesh = _preprocess(target, target_weight)
        matrix: tn.F44
        cost: float
        matrix, _transformed, cost = tm.registration.icp(
            source.vertices,
            target,
            threshold=self.threshold,
            max_iterations=self.max_iterations,
            reflection=self.reflection,
            scale=self.scale,
            translation=self.translation,
        )
        return rigid.RigidRegistrationResult(transform=matrix, cost=cost)


def _preprocess(mesh: Any, weight: tn.FN3Like | None = None) -> tm.Trimesh:
    if weight is not None:
        logger.warning("Weight is not supported, using mask instead.")
    mesh: tm.Trimesh = mkit.io.trimesh.as_trimesh(
        mkit.ops.registration.preprocess.mask_points(mesh, weight)
    )
    return mesh
