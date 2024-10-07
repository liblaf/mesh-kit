import numpy as np
import open3d as o3d
import pyvista as pv
import rich

import mkit


class Config(mkit.cli.BaseConfig):
    pass


def main(cfg: Config) -> None:
    source: pv.PolyData = mkit.ext.sculptor.get_template_maxilla()
    target: pv.PolyData = mkit.io.pyvista.load_poly_data(
        "../../01/registration/data/patients/7033231985/2016-10-18/00-skull.vtp"
    )
    source.rotate_x(180, inplace=True)
    source.translate(
        np.asarray(target.center) - np.asarray(source.center), inplace=True
    )
    source_pcd: o3d.geometry.PointCloud = mkit.io.open3d.as_point_cloud(source)
    source_pcd.normals = o3d.utility.Vector3dVector(
        source.point_normals.astype(np.float64)
    )
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1 * source.length, max_nn=100),
    )

    target_pcd: o3d.geometry.PointCloud = mkit.io.open3d.as_point_cloud(target)
    target_pcd.normals = o3d.utility.Vector3dVector(
        target.point_normals.astype(np.float64)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1 * target.length, max_nn=100),
    )

    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd,
        target_pcd,
        source_fpfh,
        target_fpfh,
        True,
        source.length * 0.1,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                source.length * 0.1
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    rich.inspect(res)
    trans = np.asarray(res.transformation)
    result = source.transform(trans, inplace=False)

    mkit.io.save(source, "data/source.vtp")
    mkit.io.save(target, "data/target.vtp")
    mkit.io.save(result, "data/result.vtp")


mkit.cli.auto_run()(main)
