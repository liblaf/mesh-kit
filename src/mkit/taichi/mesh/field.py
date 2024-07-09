from typing import Any

import taichi.lang.mesh


def place_safe(
    field: taichi.lang.mesh.MeshElementField,
    members: dict[str, Any],
    *,
    reorder: bool = False,
    needs_grad: bool = False,
) -> None:
    members = {k: v for k, v in members.items() if k not in field.field_dict}
    field.place(members, reorder=reorder, needs_grad=needs_grad)
