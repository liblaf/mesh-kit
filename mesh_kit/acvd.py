import pyacvd
import pyvista as pv
import trimesh

from mesh_kit import trimesh as _tri


def acvd(
    mesh: trimesh.Trimesh, subdivide: int = 3, cluster: int = 20000
) -> trimesh.Trimesh:
    poly_data: pv.PolyData = pv.wrap(mesh)
    clustering = pyacvd.Clustering(poly_data)
    clustering.subdivide(subdivide)
    clustering.cluster(cluster)
    return _tri.as_trimesh(clustering.create_mesh())
