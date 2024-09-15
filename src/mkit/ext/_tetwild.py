import pytetwild
import pyvista as pv

import mkit
from mkit.typing import AnySurfaceMesh


def tetwild(surface: AnySurfaceMesh) -> pv.UnstructuredGrid:
    surface: pv.PolyData = mkit.io.pyvista.as_poly_data(surface)
    tetmesh: pv.UnstructuredGrid = pytetwild.tetrahedralize_pv(surface)
    return tetmesh
