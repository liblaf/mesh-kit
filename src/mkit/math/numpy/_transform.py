import numpy as np


def inverse(mat) -> None:
    """Return inverse of square transformation matrix.

    Reference:
        1. <https://github.com/mikedh/trimesh/blob/634c608f129d15a307557800deaba954f8a30a3e/trimesh/transformations.py#L1963-L1976>
    """
    return np.linalg.inv(mat)
