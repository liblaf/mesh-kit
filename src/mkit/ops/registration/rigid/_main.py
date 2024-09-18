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
    init: nt.D44Like | None = None,
    init_global: nt.D44Like | None = None,
    inverse: bool = False,
    method: Literal["open3d", "trimesh"] = "trimesh",
    reflection: bool = False,
    scale: bool = True,
    source_weight: nt.DN3Like | None = None,
    target_weight: nt.DN3Like | None = None,
    translation: bool = True,
    **kwargs,
) -> reg.RigidRegistrationResult:
    result: reg.RigidRegistrationResult
    if inverse:
        if init is not None:
            init = tt.inverse_matrix(init)
        if init_global is not None:
            init_global = tt.inverse_matrix(init_global)
        result = rigid_registration(
            target,
            source,
            init=init,
            init_global=init_global,
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
    if init is None:
        global_result: reg.GlobalRegistrationResult = reg.global_registration(
            source, target, init=init_global
        )
        init = global_result.transform
    source: pv.PolyData = pre.downsample_mesh(source)
    init: nt.D44 = np.asarray(init)
    source = source.transform(init, inplace=False, progress_bar=True)
    target: pv.PolyData = pre.downsample_mesh(target)
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
