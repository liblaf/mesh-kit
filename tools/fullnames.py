import meshio
import open3d as o3d
import pytorch3d.structures
import pyvista as pv
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
]


def main() -> None:
    for t in TYPES:
        ic(t)


if __name__ == "__main__":
    main()
