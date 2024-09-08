from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

import mkit.typing as t

if TYPE_CHECKING:
    import meshio
    import pytorch3d.structures
    import pyvista as pv
    import trimesh as tm


class UnsupportedConversionError(ValueError):
    def __init__(self, _from: Any, to: type) -> None:
        super().__init__(f"Unsupported conversion: {type(_from)} -> {to}")


def is_meshio(mesh: Any) -> TypeGuard[meshio.Mesh]:
    return t.is_instance_named(mesh, "meshio._mesh.Mesh")


def is_polydata(mesh: Any) -> TypeGuard[pv.PolyData]:
    return t.is_instance_named(mesh, "pyvista.core.pointset.PolyData")


def is_pytorch3d(mesh: Any) -> TypeGuard[pytorch3d.structures.Meshes]:
    return t.is_instance_named(mesh, "pytorch3d.structures.meshes.Meshes")


def is_trimesh(mesh: Any) -> TypeGuard[tm.Trimesh]:
    return t.is_instance_named(mesh, "trimesh.base.Trimesh")


def is_unstructured_grid(mesh: Any) -> TypeGuard[pv.UnstructuredGrid]:
    return t.is_instance_named(mesh, "pyvista.core.pointset.UnstructuredGrid")
