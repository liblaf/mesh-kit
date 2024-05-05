from typing import Any

import taichi.lang.mesh
from loguru import logger


def place(
    field: taichi.lang.mesh.MeshElementField,
    members: dict[str, Any],
    *,
    reorder: bool = False,
    needs_grad: bool = False,
) -> None:
    members_new: dict[str, Any] = {}
    for key, dtype in members.items():
        if key in field.attr_dict:
            logger.trace("'{}' has already use as attribute name.", key)
            continue
        members_new[key] = dtype
    field.place(members_new, reorder=reorder, needs_grad=needs_grad)
