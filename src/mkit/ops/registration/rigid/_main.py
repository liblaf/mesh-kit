from typing import TYPE_CHECKING, Any

import numpy as np
import trimesh.transformations as tt

import mkit.ops.registration.preprocess as pre
import mkit.typing.numpy as nt
from mkit.ops.registration import rigid

if TYPE_CHECKING:
    import pyvista as pv


def rigid_registration(
    source: Any,
    target: Any,
    *,
    estimate_init: bool = True,
    init: nt.F44Like | None = None,
    inverse: bool = False,
    method: rigid.RigidRegistrationMethod | None = None,
    source_weight: nt.FN3Like | None = None,
    target_weight: nt.FN3Like | None = None,
) -> rigid.RigidRegistrationResult:
    result: rigid.RigidRegistrationResult
    if inverse:
        if init is not None:
            init = tt.inverse_matrix(init)
        result = rigid_registration(
            target,
            source,
            estimate_init=estimate_init,
            init=init,
            method=method,
            source_weight=source_weight,
            target_weight=target_weight,
        )
        result.transform = tt.inverse_matrix(result.transform)
        return result
    source: pv.PolyData = pre.simplify_mesh(source)
    target: pv.PolyData = pre.simplify_mesh(target)
    if init is None:
        init = tt.identity_matrix()
    if estimate_init:
        init = pre.estimate_transform(source, target, init)
    init: nt.F44 = np.asarray(init)
    source = source.transform(init)
    if method is None:
        method = rigid.TrimeshICP()
    result = method(
        source, target, source_weight=source_weight, target_weight=target_weight
    )
    result.transform = tt.concatenate_matrices(result.transform, init)
    return result
