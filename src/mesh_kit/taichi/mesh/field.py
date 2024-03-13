from typing import Any

from taichi.lang import mesh as ti_mesh


def place_safe(field: ti_mesh.MeshElementField, members: dict[str, Any]):
    members = {k: v for k, v in members.items() if k not in field.attr_dict}
    field.place(members)
