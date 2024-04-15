import meshio
import trimesh

from mkit.io import _trimesh


def to_meshio(mesh: trimesh.Trimesh) -> meshio.Mesh:
    return _trimesh.trimesh2meshio(mesh)
