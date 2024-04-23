import functools
import pathlib

import meshio
import numpy as np
import pyvista as pv
import scipy
import scipy.interpolate
import trimesh
from icecream import ic
from mkit import io as _io
from mkit import tetgen as _tet
from mkit import trimesh as _tri
from numpy import typing as npt

TEMPLATE_DIR: pathlib.Path = pathlib.Path.home() / "Documents" / "data" / "template"
mesh_fix = functools.partial(
    _tri.mesh_fix, verbose=True, joincomp=True, remove_smallest_components=True
)
face: trimesh.Trimesh = trimesh.load(TEMPLATE_DIR / "01-face.ply")
face = mesh_fix(face)
skull_old: trimesh.Trimesh = trimesh.load(TEMPLATE_DIR / "00-skull.ply")
mandible: trimesh.Trimesh
maxilla: trimesh.Trimesh
mandible, maxilla = skull_old.split()
mandible = mesh_fix(mandible)
maxilla = mesh_fix(maxilla)
skull: trimesh.Trimesh = trimesh.boolean.union([mandible, maxilla])
holes: npt.NDArray[np.float64] = np.asarray([_tri.find_inner_point(skull)])
mesh: trimesh.Trimesh = trimesh.util.concatenate([face, skull])
mesh_io: meshio.Mesh = _io.to_meshio(mesh)
mesh_io.field_data["holes"] = holes
tetra: meshio.Mesh = _tet.tetgen(mesh_io)
tetra.write("tetra.vtu")


surface = trimesh.Trimesh(tetra.points, tetra.get_cells_type("triangle"))
surface.export("surface.ply")
skull: trimesh.Trimesh
face, skull = surface.split()
disp_old: npt.NDArray[np.float64] = np.tile(
    [100.0, 0.0, 0.0], (len(skull_old.vertices), 1)
)
disp_new: npt.NDArray[np.float64] = scipy.interpolate.griddata(
    skull_old.vertices, disp_old, skull.vertices, "nearest"
)


skull_old.export("pre-skull-gt.ply")
skull_old.vertices += disp_old
skull_old.export("post-skull-gt.ply")
skull.export("pre-skull-interp.ply")
skull.vertices += disp_new
skull.export("post-skull-interp.ply")
