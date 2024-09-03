import pyvista as pv

import mkit


def cylinder() -> pv.UnstructuredGrid:
    surface: pv.PolyData = pv.Cylinder()
    tetmesh: pv.UnstructuredGrid = mkit.ext.tetwild(surface)
    return tetmesh
