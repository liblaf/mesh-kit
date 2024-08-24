import numpy as np
import numpy.typing as npt


def hess_coords(_tetra: npt.ArrayLike) -> npt.NDArray[np.integer]:
    tetra: npt.NDArray[np.integer] = np.asarray(_tetra)
    n_cells: int = len(tetra)
    coords: npt.NDArray[np.integer] = np.zeros((4, n_cells * 4 * 3 * 4 * 3), int)
    coords[0, :] = np.repeat(tetra, 3 * 4 * 3)
    coords[1, :] = np.repeat(np.tile([0, 1, 2], n_cells * 4), 4 * 3)
    coords[2, :] = np.repeat(np.tile(tetra, 12), 3)
    coords[3, :] = np.tile([0, 1, 2], n_cells * 4 * 3 * 4)
    return coords
