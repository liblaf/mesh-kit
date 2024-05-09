import meshio
import numpy as np
import pytorch3d.structures
import pyvista as pv
import trimesh
from numpy import typing as npt

from mkit._typing import StrPath
from mkit.io.types import AnyMesh


def load_trimesh(filename: StrPath) -> trimesh.Trimesh:
    """
    Args:
        filename:

    Returns:
        trimesh.Trimesh
    """
    try:
        mesh_tr: trimesh.Trimesh = trimesh.load(str(filename))
        return mesh_tr
    except Exception:
        mesh_io: meshio.Mesh = meshio.read(filename)
        return as_trimesh(mesh_io)


def as_trimesh(mesh: AnyMesh) -> trimesh.Trimesh:
    match mesh:
        case trimesh.Trimesh():
            return mesh
        case meshio.Mesh():
            return meshio_to_trimesh(mesh)
        case pytorch3d.structures.Meshes():
            raise NotImplementedError  # TODO
        case pv.PolyData():
            return pyvista_to_trimesh(mesh)
        case _:
            raise NotImplementedError(f"unsupported mesh: {mesh}")


def meshio_to_trimesh(mesh: meshio.Mesh) -> trimesh.Trimesh:
    verts: npt.NDArray[np.floating] = mesh.points
    faces: npt.NDArray[np.integer] = mesh.get_cells_type("triangle")
    return trimesh.Trimesh(verts, faces)


def pyvista_to_trimesh(mesh: pv.PolyData) -> trimesh.Trimesh:
    verts: npt.NDArray[np.floating] = mesh.points
    faces: npt.NDArray[np.integer] = mesh.regular_faces
    return trimesh.Trimesh(verts, faces)
