from typing import TYPE_CHECKING, Any

import numpy as np
import trimesh as tm
import trimesh.transformations as tf
from loguru import logger

import mkit
import mkit.ops.registration.preprocess as pre
import mkit.typing.numpy as tn
from mkit.ops.registration import rigid

if TYPE_CHECKING:
    import pyvista as pv


def rigid_registration(
    source: Any,
    target: Any,
    *,
    estimate_init: bool = True,
    init: tn.F44Like | None = None,
    inverse: bool = False,
    method: rigid.RigidRegistrationMethod | None = None,
    source_weight: tn.FN3Like | None = None,
    target_weight: tn.FN3Like | None = None,
) -> rigid.RigidRegistrationResult:
    result: rigid.RigidRegistrationResult
    if inverse:
        if init is not None:
            init = tf.inverse_matrix(init)
        result = rigid_registration(
            target,
            source,
            estimate_init=estimate_init,
            init=init,
            method=method,
            source_weight=source_weight,
            target_weight=target_weight,
        )
        result.transform = tf.inverse_matrix(result.transform)
        return result
    source: tm.Trimesh = _preprocess(source, source_weight)
    target: tm.Trimesh = _preprocess(target, target_weight)
    source_simplified: pv.PolyData = pre.simplify_mesh(source)
    target_simplified: pv.PolyData = pre.simplify_mesh(target)
    if init is None:
        init = tf.identity_matrix()
    if estimate_init:
        init = pre.estimate_transform(source_simplified, target_simplified, init)
    init: tn.F44 = np.asarray(init)
    source_simplified = source_simplified.transform(init)
    if method is None:
        method = rigid.TrimeshICP()
    result = method(source_simplified, target_simplified)
    result.transform = tf.concatenate_matrices(result.transform, init)
    return result


def _preprocess(mesh: Any, weight: tn.FN3Like | None = None) -> tm.Trimesh:
    if weight is not None:
        logger.warning("Weight is not supported, using mask instead.")
    mesh: tm.Trimesh = mkit.io.trimesh.as_trimesh(
        mkit.ops.registration.preprocess.mask_points(mesh, weight)
    )
    return mesh
