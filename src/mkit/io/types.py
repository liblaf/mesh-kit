from typing import TypeAlias

import meshio
import pytorch3d.structures
import pyvista as pv
import taichi as ti
import trimesh

AnyMesh: TypeAlias = (
    meshio.Mesh
    | pytorch3d.structures.Meshes
    | pv.PolyData
    | ti.MeshInstance
    | trimesh.Trimesh
)
