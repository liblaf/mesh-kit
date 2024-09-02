import pyvista as pv

from mkit.ext import tetwild


def cylinder() -> pv.UnstructuredGrid:
    surface: pv.PolyData = pv.Cylinder()
    tetmesh: pv.UnstructuredGrid = tetwild(surface)
    return tetmesh
