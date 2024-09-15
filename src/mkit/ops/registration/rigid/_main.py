from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import trimesh as tm

import mkit
import mkit.typing.numpy as nt
from mkit.ops.registration import RigidRegistrationResult

if TYPE_CHECKING:
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
    **kwargs,
) -> RigidRegistrationResult:
    result: RigidRegistrationResult
    if inverse:
        result = rigid_registration(target, source, method=method, **kwargs)
        result.transform = tm.transformations.inverse_matrix(result.transform)
        return result
    if init is None:
        global_result: GlobalRegistrationResult = (
            mkit.ops.registration.global_registration(source, target)
        )
        init = global_result.transform
    fn = _METHODS[method]
    result = fn(source, target, init=init, **kwargs)
    return result
