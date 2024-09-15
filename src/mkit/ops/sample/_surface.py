from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pyvista as pv
import trimesh as tm

import mkit
import mkit.typing.numpy as nt
from mkit.typing import AnySurfaceMesh


def sample_surface(
    mesh: AnySurfaceMesh,
    count: int,
    point_data: Mapping[str, npt.ArrayLike] | pv.DataSetAttributes | None = None,
) -> pv.PolyData:
    mesh: tm.Trimesh = mkit.io.trimesh.as_trimesh(mesh)
    samples: nt.DN3
    face_index: nt.IN
    samples, face_index = mesh.sample(count, return_index=True)
    sample: pv.PolyData = pv.wrap(samples)
    if point_data is not None:
        barycentric: nt.DN3 = tm.triangles.points_to_barycentric(
            mesh.triangles[face_index], samples
        )
        for k, v in point_data.items():
            data: npt.NDArray = np.asarray(v)[mesh.faces[face_index]]
            data_interp: npt.NDArray = np.einsum("ij,ij...->i...", barycentric, data)
            sample[k] = mkit.math.numpy.cast(data_interp, data.dtype)
    return sample
