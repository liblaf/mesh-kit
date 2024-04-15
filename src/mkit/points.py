from typing import Any

import numpy as np
from numpy import typing as npt
from scipy import spatial


def position_to_index(
    points: npt.ArrayLike,
    positions: npt.ArrayLike,
    *,
    distance_upper_bound: float = np.inf,
) -> npt.NDArray[np.intp]:
    """
    Args:
        points: (N, 3) float
        positions: (M, 3) float

    Returns:
        (M,) int
    """
    _: Any
    tree = spatial.KDTree(points)
    idx: npt.NDArray[np.intp]
    _, idx = tree.query(positions, distance_upper_bound=distance_upper_bound)
    return idx


def position_to_point_mask(
    points: npt.ArrayLike,
    positions: npt.ArrayLike,
    *,
    distance_upper_bound: float = np.inf,
) -> npt.NDArray[np.bool_]:
    points = np.asarray(points)
    idx: npt.NDArray[np.intp] = position_to_index(
        points, positions, distance_upper_bound=distance_upper_bound
    )
    mask: npt.NDArray[np.bool_] = np.zeros(len(points), np.bool_)
    mask[idx] = True
    return mask


def point_mask_to_face_mask(
    faces: npt.ArrayLike, point_mask: npt.ArrayLike
) -> npt.NDArray[np.bool_]:
    """
    Args:
        point_mask: (N,) bool
        faces: (M, 3) int

    Returns:
        face_mask: (M,) bool
    """
    faces = np.asarray(faces)
    point_mask = np.asarray(point_mask)
    return np.all(point_mask[faces], axis=1)


def face_mask_to_point_mask(
    num_points: int, faces: npt.ArrayLike, face_mask: npt.ArrayLike
) -> npt.NDArray[np.bool_]:
    """
    Args:
        face_mask: (M,) bool
        faces: (M, 3) int

    Returns:
        point_mask: (N,) bool
    """
    faces = np.asarray(faces)
    face_mask = np.asarray(face_mask)
    point_mask: npt.NDArray[np.bool_] = np.zeros(num_points, np.bool_)
    point_mask[faces[face_mask]] = True
    return point_mask
