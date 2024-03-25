import pathlib
from typing import Any

import meshio
import meshtaichi_patcher
import taichi as ti
from numpy import typing as npt
from taichi.lang import mesh as ti_mesh


def from_meshio(mesh: meshio.Mesh, relations: list[str] = []) -> ti.MeshInstance:
    total: dict[int, npt.NDArray] = {
        0: mesh.points,
        3: mesh.get_cells_type("tetra"),
    }
    patcher = meshtaichi_patcher.meshpatcher.MeshPatcher(total)
    patcher.patcher.patch_size = 256
    patcher.patcher.cluster_option = "greedy"
    patcher.patcher.debug = False
    patcher.patch(max_order=-1, patch_relation="all")
    meta: dict[str, Any] = patcher.get_meta(relations)
    metadata: ti_mesh.MeshMetadata = ti.Mesh.generate_meta(meta)
    return ti.Mesh._create_instance(metadata)  # pyright: ignore


def load(file: str | pathlib.Path, relations: list[str] = []) -> ti.MeshInstance:
    mesh: meshio.Mesh = meshio.read(file)
    return from_meshio(mesh, relations)
