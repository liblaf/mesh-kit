from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pyvista as pv
import trimesh as tm

import mkit
import mkit.typing as t
import mkit.typing.numpy as nt


def surface_to_surface(
    source: t.AnyTriMesh,
    target: t.AnyPointSet,
    point_data: Mapping[str, npt.ArrayLike] | pv.DataSetAttributes | None = None,
    point_data_names: Sequence[str] | None = None,
) -> pv.PolyData:
    source: pv.PolyData = mkit.io.pyvista.as_poly_data(source)
    target: pv.PolyData = mkit.io.pyvista.as_poly_data(target)
    if point_data is None:
        if point_data_names is not None:
            point_data = {k: source.point_data[k] for k in point_data_names}
        else:
            point_data = source.point_data
    source_tr: tm.Trimesh = mkit.io.trimesh.as_trimesh(source)
    target_tr: tm.Trimesh = mkit.io.trimesh.as_trimesh(target)
    closest: nt.DN3
    triangle_id: nt.IN
    closest, _dist, triangle_id = source_tr.nearest.on_surface(target_tr.vertices)
    faces: nt.IN3 = source_tr.faces[triangle_id]
    barycentric: nt.DN3 = tm.triangles.points_to_barycentric(
        source_tr.vertices[faces], closest
    )
    for k, v in point_data.items():
        data: npt.NDArray = np.asarray(v)[faces]  # (V, 3, ...)
        data_interp: npt.NDArray = np.einsum("ij,ij...->i...", barycentric, data)
        target[k] = mkit.math.numpy.cast(data_interp, data.dtype)
    return target
