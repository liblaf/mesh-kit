from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

from mkit.typing import is_instance_named

if TYPE_CHECKING:
    import meshio
    import pytorch3d.structures
    import pyvista as pv
    import trimesh


# TODO: Better typing
AnyTriMesh = Any
AnyTetMesh = Any
AnyMesh = AnyTriMesh | AnyTetMesh


class UnsupportedConversionError(ValueError):
    def __init__(self, _from: Any, to: type) -> None:
        super().__init__(f"Unsupported conversion: {type(_from)} -> {to}")


def is_meshio(mesh: Any) -> TypeGuard[meshio.Mesh]:
    return is_instance_named(mesh, "meshio._mesh.Mesh")


def is_polydata(mesh: Any) -> TypeGuard[pv.PolyData]:
    return is_instance_named(mesh, "pyvista.core.pointset.PolyData")


def is_pytorch3d(mesh: Any) -> TypeGuard[pytorch3d.structures.Meshes]:
    return is_instance_named(mesh, "pytorch3d.structures.meshes.Meshes")


def is_trimesh(mesh: Any) -> TypeGuard[trimesh.Trimesh]:
    return is_instance_named(mesh, "trimesh.base.Trimesh")


def is_unstructured_grid(mesh: Any) -> TypeGuard[pv.UnstructuredGrid]:
    return is_instance_named(mesh, "pyvista.core.pointset.UnstructuredGrid")
