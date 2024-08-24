from __future__ import annotations

from typing import Any

# TODO: Better typing
AnyTriMesh = Any
AnyTetMesh = Any
AnyMesh = AnyTriMesh | AnyTetMesh


class UnsupportedConversionError(ValueError):
    def __init__(self, from_: Any, to: type) -> None:
        super().__init__(f"Unsupported conversion: {type(from_)} -> {to}")


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
