from typing import Any

import numpy as np
import open3d as o3d

import mkit
import mkit.typing.numpy as nt
from mkit.ops.registration import RigidRegistrationResult


def icp_open3d(
    source: Any,
    target: Any,
    *,
    init: nt.D44Like | None = None,
    distance_threshold: float = 0.01,
) -> RigidRegistrationResult:
    raise NotImplementedError
    source: o3d.geometry.PointCloud = mkit.io.open3d.as_point_cloud(source)
    target: o3d.geometry.PointCloud = mkit.io.open3d.as_point_cloud(target)
    if init is not None:
        init: nt.D44 = np.asarray(init, np.float64)
    result: o3d.pipelines.registration.RegistrationResult = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return RigidRegistrationResult(
        transform=result.transformation, cost=result.inlier_rmse
    )
