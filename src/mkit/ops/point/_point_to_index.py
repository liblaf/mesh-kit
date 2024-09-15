import numpy as np
from scipy.spatial import KDTree

import mkit.typing.numpy as nt


def point_to_index(
    points: nt.DN3Like, query: nt.DN3Like, *, tol: float = 1e-6
) -> nt.IN3:
    scale: float = np.linalg.norm(np.ptp(points, axis=0))
    tree: KDTree = KDTree(points)
    index: nt.IN3
    _dist, index = tree.query(query, distance_upper_bound=tol * scale)
    return index
