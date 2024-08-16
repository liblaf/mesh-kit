from typing import Any


class UnsupportedMeshError(ValueError):
    def __init__(self, mesh: Any) -> None:
        super().__init__(f"Unsupported mesh type: {type(mesh)}")


def is_meshio(mesh: Any) -> bool:
    try:
        import meshio

        return isinstance(mesh, meshio.Mesh)
    except ImportError:
        return False


def is_polydata(mesh: Any) -> bool:
    try:
        import pyvista as pv

        return isinstance(mesh, pv.PolyData)
    except ImportError:
        return False


def is_taichi(mesh: Any) -> bool:
    try:
        import taichi as ti

        return isinstance(mesh, ti.MeshInstance)
    except ImportError:
        return False


def is_trimesh(mesh: Any) -> bool:
    try:
        import trimesh

        return isinstance(mesh, trimesh.Trimesh)
    except ImportError:
        return False


def is_unstructured_grid(mesh: Any) -> bool:
    try:
        import pyvista as pv

        return isinstance(mesh, pv.UnstructuredGrid)
    except ImportError:
        return False
