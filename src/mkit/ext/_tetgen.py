from typing import TYPE_CHECKING

import pyvista as pv
import tetgen as tg

import mkit
from mkit.io import AnyTriMesh

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


def tetgen(_surface: AnyTriMesh) -> pv.UnstructuredGrid:
    surface: pv.PolyData = mkit.io.as_polydata(_surface)
    surface.triangulate(inplace=True, progress_bar=True)
    tgen: tg.TetGen = tg.TetGen(surface)
    nodes: npt.NDArray[np.floating]
    elems: npt.NDArray[np.integer]
    nodes, elems = tgen.tetrahedralize()
    return mkit.io.make_tet_mesh(nodes, elems)
