from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import trimesh.transformations as tf

import mkit
import mkit.ops.registration as reg
import mkit.ops.registration.preprocess as pre
import mkit.typing.numpy as tn

if TYPE_CHECKING:
    import pyvista as pv

_METHODS: dict[str, Callable] = {"open3d": reg.global_.fgr_based_on_feature_matching}


def global_registration(
    source: Any,
    target: Any,
    *,
    init: tn.F44Like | None = None,
    inverse: bool = False,
    method: Literal["open3d"] = "open3d",
    normalize: bool = True,
    **kwargs,
) -> reg.GlobalRegistrationResult:
    result: reg.GlobalRegistrationResult
    if inverse:
        if init is not None:
            init = tf.inverse_matrix(init)
        result = global_registration(
            target, source, init=init, method=method, normalize=normalize, **kwargs
        )
        result.transform = tf.inverse_matrix(result.transform)
        return result
    source = pre.simplify_mesh(source)
    target = pre.simplify_mesh(target)
    if init is None:
        init = tf.identity_matrix()
    init: tn.F44 = np.asarray(init)
    source = source.transform(init, inplace=False, progress_bar=True)
    if normalize:
        source_norm: tn.F44 = mkit.ops.transform.norm_transformation(source)
        target_denorm: tn.F44 = mkit.ops.transform.denorm_transformation(target)
        source: pv.PolyData = mkit.ops.transform.normalize(source)
        target: pv.PolyData = mkit.ops.transform.normalize(target)
    else:
        source_norm = tf.identity_matrix()
        target_denorm = tf.identity_matrix()
    fn = _METHODS[method]
    result = fn(source, target, **kwargs)
    result.transform = tf.concatenate_matrices(
        target_denorm, result.transform, source_norm, init
    )
    return result
