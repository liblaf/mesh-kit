from typing import TYPE_CHECKING, Any

import numpy as np
import open3d as o3d

import mkit
from mkit.ops.registration import GlobalRegistrationResult

if TYPE_CHECKING:
    import pyvista as pv


def fgr_based_on_feature_matching(
    source: Any,
    target: Any,
    *,
    normal_radius: float = 0.04,
    normal_max_nn: int = 30,
    feature_radius: float = 0.10,
    feature_max_nn: int = 100,
    distance_threshold: float = 0.01,
) -> GlobalRegistrationResult:
    source: o3d.geometry.PointCloud
    source_fpfh: o3d.pipelines.registration.Feature
    source, source_fpfh = _preprocess(
        source,
        normal_radius=normal_radius,
        normal_max_nn=normal_max_nn,
        feature_radius=feature_radius,
        feature_max_nn=feature_max_nn,
    )
    target: o3d.geometry.PointCloud
    target_fpfh: o3d.pipelines.registration.Feature
    target, target_fpfh = _preprocess(
        target,
        normal_radius=normal_radius,
        normal_max_nn=normal_max_nn,
        feature_radius=feature_radius,
        feature_max_nn=feature_max_nn,
    )
    result: o3d.pipelines.registration.RegistrationResult = (
        o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source,
            target,
            source_fpfh,
            target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold
            ),
        )
    )
    return GlobalRegistrationResult(
        correspondence_set=result.correspondence_set,
        fitness=result.fitness,
        inlier_rmse=result.inlier_rmse,
        transform=result.transformation,
    )


def _preprocess(
    pcd: Any,
    *,
    normal_radius: float = 0.04,
    normal_max_nn: int = 30,
    feature_radius: float = 0.10,
    feature_max_nn: int = 100,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    pcd: o3d.geometry.PointCloud = mkit.io.open3d.as_point_cloud(pcd)
    try:
        mesh: pv.PolyData = mkit.io.pyvista.as_poly_data(pcd)
        pcd.normals = o3d.utility.Vector3dVector(mesh.point_normals.astype(np.float64))
    except (KeyError, RuntimeError):
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=normal_max_nn
            )
        )
    fpfh: o3d.pipelines.registration.Feature = (
        o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=feature_radius, max_nn=feature_max_nn
            ),
        )
    )
    return pcd, fpfh
