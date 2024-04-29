from typing import Unpack

import meshio

from mkit.io import _meshio
from mkit.io._meshio import as_meshio as as_meshio
from mkit.io._meshio import load_meshio as load_meshio
from mkit.io._trimesh import as_trimesh as as_trimesh
from mkit.io._trimesh import load_trimesh as load_trimesh
from mkit.io.types import AnyMesh
from mkit.typing import StrPath

__all__ = [
    "as_meshio",
    "as_trimesh",
    "load_meshio",
    "load_trimesh",
    "save",
]


def save(filename: StrPath, mesh: AnyMesh, **kwargs: Unpack[_meshio.Attrs]) -> None:
    mesh_io: meshio.Mesh = _meshio.as_meshio(mesh, **kwargs)
    mesh_io.write(filename)
