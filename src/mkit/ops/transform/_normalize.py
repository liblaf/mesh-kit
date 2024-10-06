from typing import Any

import pyvista as pv
import trimesh.transformations as tt

import mkit
import mkit.typing.numpy as nt


def normalize(
    mesh: Any,
    *,
    transform_all_input_vectors: bool = False,
    inplace: bool = False,
) -> pv.PolyData:
    mesh: pv.PolyData = mkit.io.pyvista.as_poly_data(mesh)
    matrix: nt.F44 = norm_transformation(mesh)
    mesh = mesh.transform(
        matrix, transform_all_input_vectors=transform_all_input_vectors, inplace=inplace
    )
    return mesh


def norm_transformation(mesh: Any) -> nt.F44:
    """Computes the normalization transformation matrix for a given mesh.

    This function calculates the inverse of the denormalization transformation
    matrix for the provided mesh and returns it as the normalization transformation
    matrix.

    Args:
        mesh: The mesh object for which the normalization transformation
              matrix is to be computed.

    Returns:
        The normalization transformation matrix.
    """
    matrix: nt.F44 = tt.inverse_matrix(denorm_transformation(mesh))
    return matrix


def denorm_transformation(mesh: Any) -> nt.F44:
    """Applies a denormalization transformation to the given mesh.

    This function converts the input mesh to a PyVista PolyData object,
    then computes a transformation matrix that scales and translates
    the mesh based on its length and center.

    Args:
        mesh: The input mesh to be transformed. It can be of any type
              that can be converted to a PyVista PolyData object.

    Returns:
        A 4x4 transformation matrix that scales and translates the mesh.
    """
    mesh: pv.PolyData = mkit.io.pyvista.as_poly_data(mesh)
    matrix: nt.F44 = tt.scale_and_translate(mesh.length, mesh.center)
    return matrix
