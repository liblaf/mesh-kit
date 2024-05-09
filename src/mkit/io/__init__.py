import pathlib
from typing import Unpack

import meshio
import numpy as np
from loguru import logger
from numpy import typing as npt

from mkit._typing import StrPath
from mkit.io import _meshio
from mkit.io._meshio import as_meshio as as_meshio
from mkit.io._meshio import load_meshio as load_meshio
from mkit.io._pyvista import as_pyvista as as_pyvista
from mkit.io._pyvista import load_pyvista as load_pyvista
from mkit.io._taichi import as_taichi as as_taichi
from mkit.io._taichi import load_taichi as load_taichi
from mkit.io._trimesh import as_trimesh as as_trimesh
from mkit.io._trimesh import load_trimesh as load_trimesh
from mkit.io.types import AnyMesh

__all__ = [
    "as_meshio",
    "as_pyvista",
    "as_taichi",
    "as_trimesh",
    "load_meshio",
    "load_pyvista",
    "load_taichi",
    "load_trimesh",
    "save",
]


def save(filename: StrPath, mesh: AnyMesh, **kwargs: Unpack[_meshio.Attrs]) -> None:
    mesh_io: meshio.Mesh = _meshio.as_meshio(mesh, **kwargs)
    if "landmarks" in mesh_io.field_data:
        idx: npt.NDArray[np.integer] = mesh_io.field_data["landmarks"]
        pos: npt.NDArray[np.floating] = mesh_io.points[idx]
        filename = pathlib.Path(filename)
        landmarks_file: StrPath = filename.with_suffix(".xyz")
        np.savetxt(landmarks_file, pos)
        logger.debug('saved {} landmarks to "{}"', len(idx), landmarks_file)
    mesh_io.write(filename)
