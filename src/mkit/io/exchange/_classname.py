from __future__ import annotations

import enum


class ClassName(enum.StrEnum):
    TRIMESH = "trimesh.Trimesh"
    PYVISTA_POLY_DATA = "pyvista.PolyData"
    PYVISTA_UNSTRUCTURED_GRID = "pyvista.UnstructuredGrid"
    PYVISTA_IMAGE_DATA = "pyvista.ImageData"
    PYTORCH3D_MESHES = "pytorch3d.structures.Meshes"
    MESHIO = "meshio.Mesh"
    # TODO: Add Open3D classes
