from typing import cast

from numpy.typing import NDArray
from pyvista import PolyData
from scipy.spatial import KDTree


def position_to_index(mesh: PolyData, position: NDArray, workers: int = 1) -> NDArray:
    tree: KDTree = KDTree(mesh.points)  # type: ignore
    index: NDArray
    _, index = cast(tuple[NDArray, NDArray], tree.query(position, workers=workers))
    return index
