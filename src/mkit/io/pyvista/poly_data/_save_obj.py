from __future__ import annotations

import collections
import functools
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyvista as pv

    import mkit.typing as mt
    import mkit.typing.numpy as tn


def save_obj(path: mt.StrPath, mesh: pv.PolyData) -> None:
    with Path(path).open("w") as fp:
        fprint = functools.partial(print, file=fp)
        for v in mesh.points:
            fprint("v", *v)
        if "GroupIds" in mesh.cell_data:
            group_ids: tn.IN = mesh.cell_data["GroupIds"]
            if "GroupNames" in mesh.field_data:
                group_names = mesh.field_data["GroupNames"]
            else:
                group_names = collections.defaultdict(str)
            last_group_id: int = -1
            for f, group_id in zip(mesh.irregular_faces, group_ids, strict=True):
                if group_id != last_group_id:
                    group_name: str = group_names[group_id]  # pyright: ignore[reportAssignmentType]
                    if group_name:
                        fprint("g", group_name)
                    else:
                        fprint("g")
                    last_group_id = group_id
                fprint("f", *[v + 1 for v in f])
        else:
            for f in mesh.irregular_faces:
                fprint("f", *[v + 1 for v in f])
