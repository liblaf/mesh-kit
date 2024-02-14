import os

import pyvista as pv
import tetgen
import trimesh
from loguru import logger

from mesh_kit.std import time as _time


@_time.timeit
def check(mesh: pv.PolyData | trimesh.Trimesh) -> bool:
    if not isinstance(mesh, pv.PolyData):
        mesh = pv.wrap(mesh)
    gen = tetgen.TetGen(mesh)
    try:
        gen.tetrahedralize()
        return True
    except RuntimeError as e:
        logger.trace(e)
        os.remove("_skipped.face")
        os.remove("_skipped.node")
        return False
