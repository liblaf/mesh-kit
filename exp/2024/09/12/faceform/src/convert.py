from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyvista as pv

import mkit


class Config(mkit.cli.BaseConfig):
    input: Path
    output: Path


def main(cfg: Config) -> None:
    mesh: pv.PolyData = mkit.io.pyvista.read_poly_data(cfg.input)
    with cfg.output.open("w") as f:
        for i in range(mesh.n_points):
            f.write(f"v {mesh.points[i][0]} {mesh.points[i][1]} {mesh.points[i][2]}\n")
        n_groups: int = np.max(mesh.cell_data["GroupIds"]) + 1
        group_names: Sequence[str] = mesh.field_data["GroupNames"]
        for i in range(n_groups):
            f.write(f"g {group_names[i]}\n")
            faces = mesh.regular_faces[mesh.cell_data["GroupIds"] == i]
            for face in faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1} {face[3] + 1}\n")


mkit.cli.auto_run()(main)
