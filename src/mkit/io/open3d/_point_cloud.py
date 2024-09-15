from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import open3d as o3d

import mkit.io._register as r
from mkit.io._typing import ClassName as C  # noqa: N814

if TYPE_CHECKING:
    import pyvista as pv


def as_point_cloud(mesh: Any) -> o3d.geometry.PointCloud:
    return r.convert(mesh, o3d.geometry.PointCloud)


@r.register(C.PYVISTA_POLY_DATA, C.OPEN3D_POINT_CLOUD)
def polydata_to_pointcloud(mesh: pv.PolyData) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(mesh.points.astype(np.float64))
    )
