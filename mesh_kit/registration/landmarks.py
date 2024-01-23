import pyvista as pv
import trimesh
from numpy import typing as npt
from scipy import spatial

from mesh_kit.common import testing


def position_to_index(
    mesh: trimesh.Trimesh | pv.PolyData, position: npt.NDArray, workers: int = 1
) -> npt.NDArray:
    """
    Parameters
    ---
    mesh : trimesh.Trimesh | pv.PolyData
    position : (n, 3) float
    workers : int

    Returns
    ---
    vertex_id : (n,) int
    """
    testing.assert_shape(position.shape, (-1, 3))
    match mesh:
        case trimesh.Trimesh():
            vertex_id: npt.NDArray
            _, vertex_id = mesh.nearest.vertex(position)
            return vertex_id
        case pv.PolyData():
            tree: spatial.KDTree = spatial.KDTree(mesh.points)
            index: npt.NDArray
            _, index = tree.query(position, workers=workers)
            return index
        case _:
            raise NotImplementedError()
