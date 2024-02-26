import trimesh
from numpy import typing as npt


def vert2face(mesh: trimesh.Trimesh, vert_mask: npt.NDArray) -> npt.NDArray:
    return vert_mask[mesh.faces].all(axis=1)
