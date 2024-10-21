import numpy as np

import mkit.typing.numpy as tn


def concatenate_matrices(*matrices: tn.F44Like | None) -> tn.F44:
    result: tn.F44 = np.eye(4)
    for matrix in matrices:
        if matrix is None:
            continue
        result @= matrix
    return result
