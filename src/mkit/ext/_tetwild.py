import pytetwild
import pyvista as pv

from mkit.io import AnyTriMesh, as_polydata


def tetwild(_surface: AnyTriMesh) -> pv.UnstructuredGrid:
    surface: pv.PolyData = as_polydata(_surface)
    tetmesh: pv.UnstructuredGrid = pytetwild.tetrahedralize_pv(surface)
    return tetmesh
