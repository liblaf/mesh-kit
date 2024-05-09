import pyvista as pv
import trimesh

from mkit._typing import StrPath
from mkit.io.types import AnyMesh


def load_pyvista(filename: StrPath) -> pv.PolyData:
    raise NotImplementedError  # TODO


def as_pyvista(mesh: AnyMesh) -> pv.PolyData:
    match mesh:
        case pv.PolyData():
            return mesh
        case trimesh.Trimesh():
            return trimesh_to_pyvista(mesh)
        case _:
            raise NotImplementedError(f"unsupported mesh: {mesh}")


def trimesh_to_pyvista(mesh: trimesh.Trimesh) -> pv.PolyData:
    mesh_pv: pv.PolyData = pv.wrap(mesh)
    return mesh_pv
