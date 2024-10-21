from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyvista as pv

import mkit.typing as mt
import mkit.utils as mu

if TYPE_CHECKING:
    from collections.abc import Iterable


def load_obj(fpath: mt.StrPath) -> pv.PolyData:
    path: Path = Path(fpath)
    points: list[list[float]] = []
    faces: list[list[int]] = []
    group_ids: list[int] = []
    group_names: list[str] = []
    current_group_id: int = 0
    for line in mu.strip_comments(path.read_text()):
        cmd: str
        values: list[str]
        cmd, *values = line.split()
        match cmd:
            case "v":
                points.append([float(v) for v in values])
            case "f":
                faces.append(_parse_f(values))
                if len(group_names) == 0:
                    group_names.append("")
                group_ids.append(current_group_id)
            case "g":
                if len(values) >= 1:
                    if (name := values[0]) in group_names:
                        current_group_id = group_names.index(name)
                    else:
                        group_names.append(name)
                        current_group_id = len(group_names) - 1
                else:
                    group_names.append("")
                    current_group_id = len(group_names) - 1
            case "vt" | "vn":
                # TODO: load `vt`, `vn`
                pass
            case _:
                mu.warning_once("Unknown element: {}", line)
    mesh: pv.PolyData = pv.PolyData.from_irregular_faces(points, faces)
    if len(group_names) > 1 or group_names[0] != "":
        mesh.cell_data["GroupIds"] = group_ids
        mesh.field_data["GroupNames"] = group_names
    return mesh


def _parse_f(values: Iterable[str]) -> list[int]:
    splits: list[list[str]] = [v.split("/") for v in values]
    v: list[int] = [int(s[0]) - 1 for s in splits]  # vertex indices
    # TODO: load vertex texture coordinate indices & vertex normal indices
    return v
