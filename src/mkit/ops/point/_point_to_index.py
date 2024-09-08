import numpy as np
from scipy.spatial import KDTree

import mkit.typing.numpy as n


def point_to_index(points: n.DN3Like, query: n.DN3Like, *, tol: float = 1e-6) -> n.IN3:
    scale: float = np.linalg.norm(np.ptp(points, axis=0))
    tree: KDTree = KDTree(points)
    index: n.IN3
    _dist, index = tree.query(query, distance_upper_bound=tol * scale)
    return index
