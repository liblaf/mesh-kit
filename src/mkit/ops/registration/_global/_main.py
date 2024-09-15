from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import trimesh as tm

import mkit
import mkit.ops.registration.preprocess as pre
import mkit.typing.numpy as nt
from mkit.ops.registration import GlobalRegistrationResult

if TYPE_CHECKING:
    import pyvista as pv

_METHODS: dict[str, Callable] = {
    "open3d": mkit.ops.registration._global.fgr_based_on_feature_matching  # noqa: SLF001
}


def global_registration(
    source: Any,
    target: Any,
    *,
    inverse: bool = False,
    method: Literal["open3d"] = "open3d",
    **kwargs,
) -> GlobalRegistrationResult:
    result: GlobalRegistrationResult
    if inverse:
        result = global_registration(target, source, method=method, **kwargs)
        result.transform = tm.transformations.inverse_matrix(result.transform)
        return result
    source_norm: nt.D44 = mkit.ops.transform.normalize_transform(source)
    target_denorm: nt.D44 = mkit.ops.transform.denormalize_transform(target)
    source: pv.PolyData = mkit.ops.transform.normalize(source)
    source = pre.sample_points(source)
    target: pv.PolyData = mkit.ops.transform.normalize(target)
    target = pre.sample_points(target)
    fn = _METHODS[method]
    result = fn(source, target, **kwargs)
    result.transform = tm.transformations.concatenate_matrices(
        target_denorm, result.transform, source_norm
    )
    return result
