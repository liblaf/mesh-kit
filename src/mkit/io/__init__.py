from typing import TypeAlias

import meshio
import pyvista as pv
import trimesh

from mkit.io import _pyvista, _trimesh

AnyMesh: TypeAlias = meshio.Mesh | pv.PolyData | trimesh.Trimesh


def to_meshio(mesh: AnyMesh) -> meshio.Mesh:
    match mesh:
        case meshio.Mesh():
            return mesh
        case pv.PolyData():
            return _pyvista.polydata2meshio(mesh)
        case trimesh.Trimesh():
            return _trimesh.trimesh2meshio(mesh)
        case _:
            raise ValueError("Unsupported mesh type")


def to_polydata(mesh: AnyMesh) -> pv.PolyData:
    return _pyvista.meshio2polydata(to_meshio(mesh))


def to_trimesh(mesh: AnyMesh) -> trimesh.Trimesh:
    return _trimesh.meshio2trimesh(to_meshio(mesh))
