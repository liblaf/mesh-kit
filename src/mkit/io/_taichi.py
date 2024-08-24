from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mkit.io.typing import UnsupportedConversionError, is_meshio, is_taichi

if TYPE_CHECKING:
    from collections.abc import Iterable

    import meshio
    import taichi as ti
    from numpy.typing import NDArray


def as_taichi(mesh: Any, relations: Iterable[str] = []) -> ti.MeshInstance:
    import taichi as ti

    if is_taichi(mesh):
        return mesh
    if is_meshio(mesh):
        return meshio_to_taichi(mesh, relations)
    raise UnsupportedConversionError(mesh, ti.MeshInstance)


def meshio_to_taichi(
    mesh: meshio.Mesh, relations: Iterable[str] = []
) -> ti.MeshInstance:
    import meshtaichi_patcher
    import taichi as ti
    import taichi.lang.mesh

    total: dict[int, NDArray] = {
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
    return ti.Mesh._create_instance(metadata)  # noqa: SLF001
