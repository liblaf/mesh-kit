import pytetwild
import pyvista as pv

import mkit
from mkit.io import AnyTriMesh


def tetwild(_surface: AnyTriMesh) -> pv.UnstructuredGrid:
    surface: pv.PolyData = mkit.io.as_polydata(_surface)
    tetmesh: pv.UnstructuredGrid = pytetwild.tetrahedralize_pv(surface)
    return tetmesh
