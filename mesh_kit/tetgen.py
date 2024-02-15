import os
import pathlib
import subprocess
import sys
import tempfile

import pyvista as pv
import tetgen
import trimesh
from loguru import logger

from mesh_kit.std import time as _time


@_time.timeit
def check(mesh: pv.PolyData | trimesh.Trimesh) -> bool:
    try:
        return _check_external(mesh)
    except Exception as e:
        logger.trace(e)
        return _check_internal(mesh)


def _check_internal(mesh: pv.PolyData | trimesh.Trimesh) -> bool:
    if not isinstance(mesh, pv.PolyData):
        mesh = pv.wrap(mesh)
    gen = tetgen.TetGen(mesh)
    try:
        gen.tetrahedralize(docheck=1)
        return True
    except RuntimeError as e:
        logger.trace(e)
        os.remove("_skipped.face")
        os.remove("_skipped.node")
        return False


def _check_external(mesh: pv.PolyData | trimesh.Trimesh) -> bool:
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = pathlib.Path(_tmpdir)
        match mesh:
            case pv.PolyData():
                filename: pathlib.Path = tmpdir / "mesh.ply"
                mesh.save(filename, binary=False)
            case trimesh.Trimesh():
                filename: pathlib.Path = tmpdir / "mesh.ply"
                mesh.export(filename, encoding="ascii")
            case _:
                raise NotImplementedError()
        popen = subprocess.Popen(
            args=["tetgen", "-N", "-E", "-F", "-C", filename],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
        )
        for line in popen.stdout:
            line: str = line.rstrip()
            if "Warning" in line:
                logger.warning(line.removeprefix("Warning:").strip())
                logger.warning(next(popen.stdout).rstrip())
                logger.warning(next(popen.stdout).rstrip())
                popen.terminate()
                return False
        return popen.wait() == 0
