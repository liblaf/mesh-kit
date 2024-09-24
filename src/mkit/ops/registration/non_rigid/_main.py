from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import pyvista as pv

import mkit
import mkit.typing.numpy as nt
from mkit.ops.registration.non_rigid import amberg_pytorch3d

if TYPE_CHECKING:
    from mkit.ops.registration.non_rigid.amberg_pytorch3d._result import (
        NonRigidRegistrationResult,
    )

_METHODS: dict[str, Callable] = {"amberg": amberg_pytorch3d.nricp_amberg_pytorch3d}


def non_rigid_registration(
    source: Any,
    target: Any,
    method: Literal["amberg"] = "amberg",
    params: amberg_pytorch3d.ParamsDict | None = None,
) -> pv.PolyData:
    source: pv.PolyData = mkit.io.pyvista.as_poly_data(source)
    target: pv.PolyData = mkit.io.pyvista.as_poly_data(target)
    target_norm: nt.F44 = mkit.ops.transform.normalize_transform(target)
    target_denorm: nt.F44 = mkit.ops.transform.denormalize_transform(target)
    source = source.transform(target_norm, inplace=False)
    target = target.transform(target_norm, inplace=False)
    fn = _METHODS[method]
    result: NonRigidRegistrationResult = fn(source, target, params)
    source.points = result.result
    source = source.transform(target_denorm, inplace=False)
    return source
