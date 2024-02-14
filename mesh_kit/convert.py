import pyvista as pv
import trimesh


def polydata2trimesh(mesh: pv.PolyData) -> trimesh.Trimesh:
    mesh = mesh.triangulate()
    return trimesh.Trimesh(
        vertices=mesh.points, faces=mesh.faces.reshape((-1, 4))[:, 1:4]
    )
