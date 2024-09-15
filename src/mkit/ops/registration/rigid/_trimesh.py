from typing import Any

import trimesh as tm

import mkit
import mkit.typing.numpy as nt
from mkit.ops.registration import RigidRegistrationResult


def icp_trimesh(
    source: Any,
    target: Any,
    *,
    init: nt.D44Like | None = None,
    threshold: float = 1e-6,
    max_iterations: int = 100,
    reflection: bool = True,
    scale: bool = True,
    translation: bool = True,
) -> RigidRegistrationResult:
    source: tm.Trimesh = mkit.io.trimesh.as_trimesh(source)
    target: tm.Trimesh = mkit.io.trimesh.as_trimesh(target)
    matrix: nt.D44
    cost: float
    matrix, _transformed, cost = tm.registration.icp(
        source.vertices,
        target,
        initial=init,
        threshold=threshold,
        max_iterations=max_iterations,
        reflection=reflection,
        scale=scale,
        translation=translation,
    )
    return RigidRegistrationResult(transform=matrix, cost=cost)
