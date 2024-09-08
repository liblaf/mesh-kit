from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pyvista as pv
import trimesh as tm

import mkit
import mkit.typing as t
import mkit.typing.numpy as n


def surface_to_surface(
    _source: t.AnyTriMesh,
    _target: t.AnyPointSet,
    point_data_names: Sequence[str] | None = None,
) -> pv.PolyData:
    source: pv.PolyData = mkit.io.as_polydata(_source)
    target: pv.PolyData = mkit.io.as_polydata(_target)
    if point_data_names is None:
        point_data_names = source.point_data.keys()
    source_tr: tm.Trimesh = mkit.io.as_trimesh(_source)
    target_tr: tm.Trimesh = mkit.io.as_trimesh(_target)
    closest: n.DN3
    triangle_id: n.IN
    closest, _, triangle_id = source_tr.nearest.on_surface(target_tr.vertices)
    faces: n.IN3 = source_tr.faces[triangle_id]
    barycentric: n.DN3 = tm.triangles.points_to_barycentric(
        source_tr.vertices[faces], closest
    )
    for k in point_data_names:
        data: npt.NDArray = source.point_data[k][faces]  # (V, 3, ...)
        data_interp: npt.NDArray = np.einsum("ij,ij...->i...", barycentric, data)
        target[k] = mkit.math.numpy.cast(data_interp, data.dtype)
    return target
