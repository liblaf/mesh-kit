import meshio
import trimesh


def from_trimesh(mesh_tr: trimesh.Trimesh) -> meshio.Mesh:
    return meshio.Mesh(points=mesh_tr.vertices, cells={"triangle": mesh_tr.faces})
