from typing import TYPE_CHECKING

import pyvista as pv
import tetgen as tg

import mkit
from mkit.typing import AnySurfaceMesh

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


def tetgen(surface: AnySurfaceMesh) -> pv.UnstructuredGrid:
    surface: pv.PolyData = mkit.io.pyvista.as_poly_data(surface)
    surface.triangulate(inplace=True, progress_bar=True)
    tgen: tg.TetGen = tg.TetGen(surface)
    nodes: npt.NDArray[np.floating]
    elems: npt.NDArray[np.integer]
    nodes, elems = tgen.tetrahedralize()
    return mkit.io.pyvista.make_tet_mesh(nodes, elems)
