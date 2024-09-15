import pyvista as pv
import trimesh as tm

import mkit
import mkit.typing.numpy as nt
from mkit.typing import AnySurfaceMesh


def normalize(
    mesh: AnySurfaceMesh,
    *,
    transform_all_input_vectors: bool = False,
    inplace: bool = False,
    progress_bar: bool = True,
) -> pv.PolyData:
    mesh: pv.PolyData = mkit.io.pyvista.as_poly_data(mesh)
    matrix: nt.D44 = normalize_transform(mesh)
    mesh = mesh.transform(
        matrix,
        transform_all_input_vectors=transform_all_input_vectors,
        inplace=inplace,
        progress_bar=progress_bar,
    )
    return mesh


def normalize_transform(mesh: AnySurfaceMesh) -> nt.D44:
    matrix: nt.D44 = tm.transformations.inverse_matrix(denormalize_transform(mesh))
    return matrix


def denormalize_transform(mesh: AnySurfaceMesh) -> nt.D44:
    mesh: pv.PolyData = mkit.io.pyvista.as_poly_data(mesh)
    matrix: nt.D44 = tm.transformations.scale_and_translate(mesh.length, mesh.center)
    return matrix
