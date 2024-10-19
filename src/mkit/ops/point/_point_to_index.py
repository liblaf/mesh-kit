import numpy as np
from scipy.spatial import KDTree

import mkit.typing.numpy as tn


def point_to_index(
    points: tn.FN3Like, query: tn.FN3Like, *, tol: float = 1e-6
) -> tn.IN3:
    scale: float = np.linalg.norm(np.ptp(points, axis=0))
    tree: KDTree = KDTree(points)
    index: tn.IN3
    _dist, index = tree.query(query, distance_upper_bound=tol * scale)
    return index
