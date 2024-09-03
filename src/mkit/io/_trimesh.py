import trimesh as tm

import mkit.io._typing as t


def as_trimesh(mesh: t.AnyTriMesh) -> tm.Trimesh:
    if t.is_trimesh(mesh):
        return mesh
    if t.is_meshio(mesh):
        return tm.Trimesh(mesh.points, mesh.get_cells_type("triangle"))
    if t.is_polydata(mesh):
        return tm.Trimesh(mesh.points, mesh.regular_faces)
    raise t.UnsupportedConversionError(mesh, tm.Trimesh)
