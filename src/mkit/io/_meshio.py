from typing import TypedDict, Unpack

import meshio
import numpy as np
import pytorch3d.structures
import pyvista as pv
import taichi as ti
import torch
import trimesh
from numpy import typing as npt

from mkit._typing import StrPath
from mkit.io.types import AnyMesh


class Attrs(TypedDict, total=False):
    point_data: dict[str, npt.ArrayLike] | None
    cell_data: dict[str, list[npt.ArrayLike]] | None
    field_data: dict[str, npt.ArrayLike] | None
    point_sets: dict[str, npt.ArrayLike] | None
    cell_sets: dict[str, list[npt.ArrayLike]] | None
    gmsh_periodic: None
    info: None


def load_meshio(filename: StrPath) -> meshio.Mesh:
    return meshio.read(filename)


def as_meshio(mesh: AnyMesh, **kwargs: Unpack[Attrs]) -> meshio.Mesh:
    match mesh:
        case meshio.Mesh():
            if point_data := kwargs.get("point_data"):
                mesh.point_data.update(point_data)
            if cell_data := kwargs.get("cell_data"):
                mesh.cell_data.update(cell_data)
            if field_data := kwargs.get("field_data"):
                mesh.field_data.update(field_data)
            return mesh
        case pytorch3d.structures.Meshes():
            return pytorch3d_to_meshio(mesh, **kwargs)
        case pv.PolyData():
            raise NotImplementedError  # TODO
        case ti.MeshInstance():
            raise NotImplementedError  # TODO
        case trimesh.Trimesh():
            return trimesh_to_meshio(mesh, **kwargs)
        case _:
            raise NotImplementedError(f"unsupported mesh: {mesh}")


def pytorch3d_to_meshio(
    mesh: pytorch3d.structures.Meshes, **kwargs: Unpack[Attrs]
) -> meshio.Mesh:
    verts_pt: torch.Tensor = mesh.verts_packed()
    faces_pt: torch.Tensor = mesh.faces_packed()
    verts_np: npt.NDArray[np.floating] = verts_pt.detach().cpu().numpy()
    faces_np: npt.NDArray[np.integer] = faces_pt.detach().cpu().numpy()
    mesh_io = meshio.Mesh(verts_np, [("triangle", faces_np)], **kwargs)
    return mesh_io


def trimesh_to_meshio(mesh: trimesh.Trimesh, **kwargs: Unpack[Attrs]) -> meshio.Mesh:
    mesh_io = meshio.Mesh(mesh.vertices, [("triangle", mesh.faces)], **kwargs)
    return mesh_io
