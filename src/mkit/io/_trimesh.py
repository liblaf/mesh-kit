import meshio
import trimesh


def trimesh2meshio(mesh: trimesh.Trimesh) -> meshio.Mesh:
    return meshio.Mesh(mesh.vertices, cells=[("triangle", mesh.faces)])
