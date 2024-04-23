import meshio
import trimesh


def trimesh2meshio(mesh: trimesh.Trimesh) -> meshio.Mesh:
    return meshio.Mesh(mesh.vertices, [("triangle", mesh.faces)])


def meshio2trimesh(mesh: meshio.Mesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(mesh.points, mesh.get_cells_type("triangle"))
