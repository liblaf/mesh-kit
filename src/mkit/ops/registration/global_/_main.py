from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import trimesh.transformations as tt

import mkit
import mkit.ops.registration as reg
import mkit.ops.registration.preprocess as pre
import mkit.typing.numpy as nt

if TYPE_CHECKING:
    import pyvista as pv

_METHODS: dict[str, Callable] = {"open3d": reg.global_.fgr_based_on_feature_matching}


def global_registration(
    source: Any,
    target: Any,
    *,
    init: nt.F44Like | None = None,
    inverse: bool = False,
    method: Literal["open3d"] = "open3d",
    normalize: bool = True,
    **kwargs,
) -> reg.GlobalRegistrationResult:
    result: reg.GlobalRegistrationResult
    if inverse:
        if init is not None:
            init = tt.inverse_matrix(init)
        result = global_registration(
            target, source, init=init, method=method, normalize=normalize, **kwargs
        )
        result.transform = tt.inverse_matrix(result.transform)
        return result
    source = pre.downsample_mesh(source)
    target = pre.downsample_mesh(target)
    if init is None:
        init = tt.identity_matrix()
    init: nt.F44 = np.asarray(init)
    source = source.transform(init, inplace=False, progress_bar=True)
    if normalize:
        source_norm: nt.F44 = mkit.ops.transform.normalize_transform(source)
        target_denorm: nt.F44 = mkit.ops.transform.denormalize_transform(target)
        source: pv.PolyData = mkit.ops.transform.normalize(source)
        target: pv.PolyData = mkit.ops.transform.normalize(target)
    else:
        source_norm = tt.identity_matrix()
        target_denorm = tt.identity_matrix()
    fn = _METHODS[method]
    result = fn(source, target, **kwargs)
    result.transform = tt.concatenate_matrices(
        target_denorm, result.transform, source_norm, init
    )
    return result
