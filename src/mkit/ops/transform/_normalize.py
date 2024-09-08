import pyvista as pv
import trimesh as tm

import mkit
import mkit.typing.numpy as n
from mkit.typing import AnySurfaceMesh


def normalize(
    _mesh: AnySurfaceMesh,
    *,
    transform_all_input_vectors: bool = False,
    progress_bar: bool = True,
) -> pv.PolyData:
    mesh: pv.PolyData = mkit.io.as_polydata(_mesh)
    matrix: n.D44 = normalize_transform(_mesh)
    mesh = mesh.transform(
        matrix,
        transform_all_input_vectors=transform_all_input_vectors,
        inplace=False,
        progress_bar=progress_bar,
    )
    return mesh


def normalize_transform(_mesh: AnySurfaceMesh) -> n.D44:
    matrix: n.D44 = tm.transformations.inverse_matrix(denormalize_transform(_mesh))
    return matrix


def denormalize_transform(_mesh: AnySurfaceMesh) -> n.D44:
    mesh: pv.PolyData = mkit.io.as_polydata(_mesh)
    matrix: n.D44 = tm.transformations.scale_and_translate(mesh.length, mesh.center)
    return matrix
