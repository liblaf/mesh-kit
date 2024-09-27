from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import trimesh.transformations as tt

import mkit.ops.registration as reg
import mkit.ops.registration.preprocess as pre
import mkit.typing.numpy as nt

if TYPE_CHECKING:
    import pyvista as pv


_METHODS: dict[str, Callable] = {
    "open3d": reg.rigid.icp_open3d,
    "trimesh": reg.rigid.icp_trimesh,
}


def rigid_registration(
    source: Any,
    target: Any,
    *,
    estimate_init: bool = True,
    init: nt.F44Like | None = None,
    inverse: bool = False,
    method: Literal["open3d", "trimesh"] = "trimesh",
    reflection: bool = False,
    scale: bool = True,
    source_weight: nt.FN3Like | None = None,
    target_weight: nt.FN3Like | None = None,
    translation: bool = True,
    **kwargs,
) -> reg.RigidRegistrationResult:
    result: reg.RigidRegistrationResult
    if inverse:
        if init is not None:
            init = tt.inverse_matrix(init)
        result = rigid_registration(
            target,
            source,
            init=init,
            method=method,
            reflection=reflection,
            scale=scale,
            source_weight=source_weight,
            target_weight=target_weight,
            translation=translation,
            **kwargs,
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
    fn = _METHODS[method]
    result = fn(
        source,
        target,
        reflection=reflection,
        scale=scale,
        source_weight=source_weight,
        target_weight=target_weight,
        translation=translation,
        **kwargs,
    )
    result.transform = tt.concatenate_matrices(result.transform, init)
    return result
