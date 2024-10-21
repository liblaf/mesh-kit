from __future__ import annotations

import enum


class ClassName(enum.StrEnum):
    MESHIO_MESH = "meshio.Mesh"
    PYTORCH3D_MESHES = "pytorch3d.structures.Meshes"
    PYVISTA_IMAGE_DATA = "pyvista.ImageData"
    PYVISTA_POLY_DATA = "pyvista.PolyData"
    PYVISTA_UNSTRUCTURED_GRID = "pyvista.UnstructuredGrid"
    TRIMESH_TRIMESH = "trimesh.Trimesh"
