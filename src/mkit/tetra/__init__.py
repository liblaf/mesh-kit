import igl
import numpy as np
from numpy import typing as npt


def boundary_faces(tetra: npt.ArrayLike) -> npt.NDArray[np.integer]:
    """
    Determine boundary faces of tetrahedra.

    Args:
        tetra: (M, 4) int: tetrahedron index list, where M is the number of tetrahedra.

    Returns:
        triangle: (N, 3) int: list of boundary faces, where N is the number of boundary faces.
    """
    tetra = np.asarray(tetra)
    triangle: npt.NDArray[np.integer] = np.asarray(igl.boundary_facets(tetra))  # pyright: ignore [reportAttributeAccessIssue]
    return triangle
