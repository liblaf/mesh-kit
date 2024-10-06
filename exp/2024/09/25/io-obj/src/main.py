from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyvista as pv
from loguru import logger

import mkit
import mkit.typing.numpy as nt


class Config(mkit.cli.BaseConfig):
    fpath: Path


def main(cfg: Config) -> None:
    points: list[list[float]] = []
    faces: list[list[int]] = []
    group_ids: list[int] = []
    group_names: list[str] = []
    current_group_id: int = 0
    for line in mkit.utils.strip_comments(cfg.fpath.read_text()):
        cmd: str
        values: list[str]
        cmd, *values = line.split()
        match cmd:
            case "v":
                points.append([float(v) for v in values])
            case "f":
                faces.append(parse_f(values))
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
                logger.warning("Unsupported element: {}", line)
            case _:
                logger.warning("Unknown element: {}", line)
    mesh: pv.PolyData = pv.PolyData.from_irregular_faces(points, faces)
    if len(group_names) > 1 or group_names[0] != "":
        mesh.cell_data["GroupIds"] = group_ids
        mesh.field_data["GroupNames"] = group_names
    ic(np.unique(mesh.cell_data["GroupIds"]))
    ic(mesh.field_data["GroupNames"])
    write(mesh)


def write(mesh: pv.PolyData) -> None:
    with open("mesh.obj", "w") as fp:
        for v in mesh.points:
            print("v", *v, file=fp)
        if "GroupIds" in mesh.cell_data:
            group_ids: nt.IN = mesh.cell_data["GroupIds"]
            group_names: npt.NDArray[np.str_]
            if "GroupNames" in mesh.field_data:
                group_names = mesh.field_data["GroupNames"]
            else:
                group_names = np.full((len(np.unique(group_ids)),), "")
            last_group_id: int = -1
            for f, group_id in zip(mesh.irregular_faces, group_ids, strict=True):
                if group_id != last_group_id:
                    group_name: str = group_names[group_id]
                    if group_name:
                        print("g", group_name, file=fp)
                    else:
                        print("g", file=fp)
                    last_group_id = group_id
                print("f", *[v + 1 for v in f], file=fp)
        else:
            for f in mesh.irregular_faces:
                print("f", *[v + 1 for v in f], file=fp)


def parse_f(values: list[str]) -> list[int]:
    splits: list[list[str]] = [v.split("/") for v in values]
    v: list[int] = [int(s[0]) - 1 for s in splits]  # vertex indices
    # TODO: load vertex texture coordinate indices & vertex normal indices
    return v


mkit.cli.auto_run()(main)
