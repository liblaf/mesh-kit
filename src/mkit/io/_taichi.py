from collections.abc import Iterable
from typing import Any

import meshio
import meshtaichi_patcher
import taichi as ti
import taichi.lang.mesh
from numpy import typing as npt

from mkit._typing import StrPath
from mkit.io.types import AnyMesh


def load_taichi(filename: StrPath) -> ti.MeshInstance:
    raise NotImplementedError  # TODO


def as_taichi(mesh: AnyMesh, relations: Iterable[str] = []) -> ti.MeshInstance:
    match mesh:
        case ti.MeshInstance():
            return mesh
        case meshio.Mesh():
            return meshio_to_taichi(mesh, relations)
        case _:
            raise NotImplementedError(f"unsupported mesh: {mesh}")


def meshio_to_taichi(
    mesh: meshio.Mesh, relations: Iterable[str] = []
) -> ti.MeshInstance:
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
    metadata: taichi.lang.mesh.MeshMetadata = ti.Mesh.generate_meta(meta)
    return ti.Mesh._create_instance(metadata)
