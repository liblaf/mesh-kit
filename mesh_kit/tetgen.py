import pathlib
import subprocess
import sys
import tempfile

import pyvista as pv
import tetgen
import trimesh
from loguru import logger

from mesh_kit import log as _log
from mesh_kit import polydata as _poly


@_log.timeit
def check(mesh: pv.PolyData | trimesh.Trimesh) -> bool:
    try:
        return _check_external(mesh)
    except FileNotFoundError as e:
        logger.trace(e)
        return _check_internal(mesh)


def _check_internal(mesh: pv.PolyData | trimesh.Trimesh) -> bool:
    if not isinstance(mesh, pv.PolyData):
        mesh = _poly.as_polydata(mesh)
    gen = tetgen.TetGen(mesh)
    try:
        gen.tetrahedralize(docheck=1)
    except RuntimeError:
        logger.exception("")
        pathlib.Path("_skipped.face").unlink(missing_ok=True)
        pathlib.Path("_skipped.node").unlink(missing_ok=True)
        return False
    else:
        return True


def _check_external(mesh: pv.PolyData | trimesh.Trimesh) -> bool:
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = pathlib.Path(_tmpdir)
        match mesh:
            case pv.PolyData():
                filename = tmpdir / "mesh.ply"
                mesh.save(filename, binary=False)
            case trimesh.Trimesh():
                filename = tmpdir / "mesh.ply"
                mesh.export(filename, encoding="ascii")
            case _:
                raise NotImplementedError
        popen = subprocess.Popen(
            args=["tetgen", "-N", "-E", "-F", "-C", filename],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
        )
        assert popen.stdout
        for line in popen.stdout:
            line = line.rstrip()  # noqa: PLW2901
            if "Warning" in line:
                logger.warning(line.removeprefix("Warning:").strip())
                logger.warning(next(popen.stdout).rstrip())
                logger.warning(next(popen.stdout).rstrip())
                popen.terminate()
                return False
        return popen.wait() == 0
