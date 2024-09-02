import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree


def point_to_index(
    points: npt.ArrayLike, query: npt.ArrayLike, *, tol: float = 1e-6
) -> npt.NDArray[np.integer]:
    scale: float = np.linalg.norm(np.ptp(points, axis=0))
    tree: KDTree = KDTree(points)
    distance: npt.NDArray[np.floating]
    index: npt.NDArray[np.integer]
    distance, index = tree.query(query, distance_upper_bound=tol * scale)
    return index
