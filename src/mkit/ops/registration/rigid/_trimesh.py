from typing import Any

import trimesh as tm
from loguru import logger

import mkit
import mkit.typing.numpy as nt
from mkit.ops.registration import RigidRegistrationResult


def icp_trimesh(
    source: Any,
    target: Any,
    *,
    max_iterations: int = 100,
    reflection: bool = False,
    scale: bool = True,
    source_weight: nt.DN3Like | None = None,
    target_weight: nt.DN3Like | None = None,
    threshold: float = 1e-6,
    translation: bool = True,
) -> RigidRegistrationResult:
    source: tm.Trimesh = _preprocess(source, source_weight)
    target: tm.Trimesh = _preprocess(target, target_weight)
    matrix: nt.D44
    cost: float
    matrix, _transformed, cost = tm.registration.icp(
        source.vertices,
        target,
        threshold=threshold,
        max_iterations=max_iterations,
        reflection=reflection,
        scale=scale,
        translation=translation,
    )
    return RigidRegistrationResult(transform=matrix, cost=cost)


def _preprocess(mesh: Any, weight: nt.DN3Like | None = None) -> tm.Trimesh:
    if weight is not None:
        logger.warning("Weight is not supported, using mask instead.")
    mesh: tm.Trimesh = mkit.io.trimesh.as_trimesh(
        mkit.ops.registration.preprocess.mask_points(mesh, weight)
    )
    return mesh
