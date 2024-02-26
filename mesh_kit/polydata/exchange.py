import pyvista as pv
import trimesh

from mesh_kit.typing import check_type as _check_type


def as_polydata(obj) -> pv.PolyData:
    match obj:
        case trimesh.Trimesh():
            return _check_type(pv.wrap(obj), pv.PolyData)
        case _:
            raise NotImplementedError
