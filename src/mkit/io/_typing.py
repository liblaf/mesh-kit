from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeGuard

import mkit.typing as t

if TYPE_CHECKING:
    import meshio
    import pytorch3d.structures
    import pyvista as pv
    import trimesh as tm


class ClassName(StrEnum):
    MESHIO = "meshio._mesh.Mesh"
    OPEN3D_POINT_CLOUD = "open3d.geometry.PointCloud"
    PYTORCH3D = "pytorch3d.structures.meshes.Meshes"
    PYVISTA_IMAGE_DATA = "pyvista.core.grid.ImageData"
    PYVISTA_POLY_DATA = "pyvista.core.pointset.PolyData"
    PYVISTA_UNSTRUCTURED_GRID = "pyvista.core.pointset.UnstructuredGrid"
    TRIMESH = "trimesh.base.Trimesh"


class UnsupportedConversionError(ValueError):
    def __init__(self, from_: Any, to: type) -> None:
        super().__init__(f"Unsupported conversion: {type(from_)} -> {to}")


def is_meshio(mesh: Any) -> TypeGuard[meshio.Mesh]:
    return t.is_instance_named(mesh, ClassName.MESHIO)


def is_pytorch3d(mesh: Any) -> TypeGuard[pytorch3d.structures.Meshes]:
    return t.is_instance_named(mesh, ClassName.PYTORCH3D)


def is_image_data(mesh: Any) -> TypeGuard[pv.ImageData]:
    return t.is_instance_named(mesh, ClassName.PYVISTA_POLY_DATA)


def is_poly_data(mesh: Any) -> TypeGuard[pv.PolyData]:
    return t.is_instance_named(mesh, ClassName.PYVISTA_POLY_DATA)


def is_unstructured_grid(mesh: Any) -> TypeGuard[pv.UnstructuredGrid]:
    return t.is_instance_named(mesh, ClassName.PYVISTA_UNSTRUCTURED_GRID)


def is_trimesh(mesh: Any) -> TypeGuard[tm.Trimesh]:
    return t.is_instance_named(mesh, ClassName.TRIMESH)
