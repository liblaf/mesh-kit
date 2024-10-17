from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import open3d as o3d

from mkit.io._register import REGISTRY
from mkit.io._typing import ClassName as C  # noqa: N814

if TYPE_CHECKING:
    import pyvista as pv

    import mkit.typing.numpy as tn


def as_point_cloud(mesh: Any) -> o3d.geometry.PointCloud:
    return REGISTRY.convert(mesh, o3d.geometry.PointCloud)


@REGISTRY.register(C.ARRAY_LIKE, C.OPEN3D_POINT_CLOUD, priority=-10)
def array_to_point_cloud(points: tn.FN3Like) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray(points, np.float64))
    )


@REGISTRY.register(C.PYVISTA_POLY_DATA, C.OPEN3D_POINT_CLOUD)
def poly_data_to_point_cloud(mesh: pv.PolyData) -> o3d.geometry.PointCloud:
    return array_to_point_cloud(mesh.points)
