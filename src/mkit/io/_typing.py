from enum import StrEnum
from typing import Any

import mkit


class ClassName(StrEnum):
    ARRAY_LIKE = "ArrayLike"
    MESHIO = "meshio._mesh.Mesh"
    OPEN3D_POINT_CLOUD = "open3d.geometry.PointCloud"
    PYTORCH3D_MESHES = "pytorch3d.structures.meshes.Meshes"
    PYVISTA_IMAGE_DATA = "pyvista.core.grid.ImageData"
    PYVISTA_POLY_DATA = "pyvista.core.pointset.PolyData"
    PYVISTA_UNSTRUCTURED_GRID = "pyvista.core.pointset.UnstructuredGrid"
    TRIMESH = "trimesh.base.Trimesh"


class UnsupportedConversionError(ValueError):
    def __init__(self, from_: Any, to: type) -> None:
        super().__init__(f"Unsupported conversion: {type(from_)} -> {to}")


def is_sub_type(obj: Any, name: str) -> bool:
    match name:
        case ClassName.ARRAY_LIKE:
            return mkit.typing.is_array_like(obj)
        case _:
            return mkit.typing.is_named_partial(obj, name)
