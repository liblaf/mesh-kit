import meshio
import pyvista as pv


def polydata2meshio(mesh: pv.PolyData) -> meshio.Mesh:
    return meshio.Mesh(mesh.points, [("triangle", mesh.regular_faces)])


def meshio2polydata(mesh: meshio.Mesh) -> pv.PolyData:
    return pv.make_tri_mesh(mesh.points, mesh.get_cells_type("triangle"))
