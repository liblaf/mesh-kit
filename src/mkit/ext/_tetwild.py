import pytetwild
import pyvista as pv

import mkit
from mkit.typing import AnySurfaceMesh


def tetwild(_surface: AnySurfaceMesh) -> pv.UnstructuredGrid:
    surface: pv.PolyData = mkit.io.as_polydata(_surface)
    tetmesh: pv.UnstructuredGrid = pytetwild.tetrahedralize_pv(surface)
    return tetmesh
