from typing import Any

import numpy as np
import numpy.typing as npt


def point_to_index(
    points: npt.ArrayLike, query: npt.ArrayLike, tol: float = 1e-6
) -> npt.NDArray[np.int32]:
    import scipy.spatial

    _: Any
    scale: float = np.linalg.norm(np.ptp(points, axis=0))  # pyright: ignore [reportAssignmentType]
    tree = scipy.spatial.KDTree(points)
    index: npt.NDArray[np.int32]
    _, index = tree.query(query, distance_upper_bound=tol * scale)
    return index
