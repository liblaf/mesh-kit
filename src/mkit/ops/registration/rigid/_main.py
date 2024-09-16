from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import trimesh as tm

import mkit
import mkit.typing.numpy as nt
from mkit.ops.registration import RigidRegistrationResult

if TYPE_CHECKING:
    import pyvista as pv

    from mkit.ops.registration import GlobalRegistrationResult

_METHODS: dict[str, Callable] = {
    "open3d": mkit.ops.registration.rigid.icp_open3d,
    "trimesh": mkit.ops.registration.rigid.icp_trimesh,
}


def rigid_registration(
    source: Any,
    target: Any,
    *,
    init: nt.D44Like | None = None,
    inverse: bool = False,
    method: Literal["open3d", "trimesh"] = "trimesh",
    reflection: bool = False,
    scale: bool = True,
    source_weight: nt.DN3Like | None = None,
    target_weight: nt.DN3Like | None = None,
    translation: bool = True,
    **kwargs,
) -> RigidRegistrationResult:
    result: RigidRegistrationResult
    if inverse:
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
        result.transform = tm.transformations.inverse_matrix(result.transform)
        return result
    if init is None:
        global_result: GlobalRegistrationResult = (
            mkit.ops.registration.global_registration(source, target)
        )
        init = global_result.transform
    source: pv.PolyData = mkit.io.pyvista.as_poly_data(source)
    init: nt.D44 = np.asarray(init)
    source = source.transform(init, inplace=False, progress_bar=True)
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
    result.transform = tm.transformations.concatenate_matrices(result.transform, init)
    return result
