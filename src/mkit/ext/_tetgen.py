from typing import TYPE_CHECKING, Any

import meshio

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pyvista as pv


def tetgen(surface: Any, *args, **kwargs) -> meshio.Mesh:
    import tetgen

    import mkit.io

    surface_mesh: pv.PolyData = mkit.io.as_polydata(surface)
    tet = tetgen.TetGen(surface_mesh)
    points: npt.NDArray[np.floating]
    tetra: npt.NDArray[np.integer]
    points, tetra = tet.tetrahedralize(*args, **kwargs)
    tetmesh = meshio.Mesh(points, [("tetra", tetra)])
    return tetmesh
