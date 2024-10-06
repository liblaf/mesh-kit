import jax
import meshio
import numpy as np
import open3d as o3d
import pytorch3d.structures
import pyvista as pv
import torch
import trimesh as tm

TYPES: list[type] = [
    meshio.Mesh,
    o3d.geometry.PointCloud,
    o3d.t.geometry.PointCloud,
    pytorch3d.structures.Meshes,
    pv.ImageData,
    pv.PolyData,
    pv.UnstructuredGrid,
    tm.Trimesh,
    jax.Array,
    np.ndarray,
    torch.Tensor,
]


def main() -> None:
    for t in TYPES:
        ic(t)


if __name__ == "__main__":
    main()
