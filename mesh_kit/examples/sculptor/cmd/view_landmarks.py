from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import trimesh
from numpy.typing import NDArray
from pyvista.plotting.plotter import Plotter
from trimesh import Trimesh
from typer import Argument

from mesh_kit.common.cli import run
from mesh_kit.common.path import landmarks_filepath

COLORS: Sequence[str] = ["green", "red"]


def main(
    mesh_filepath: Annotated[list[Path], Argument(exists=True, dir_okay=False)]
) -> None:
    plotter: Plotter = Plotter()
    for i, filepath in enumerate(mesh_filepath):
        mesh: Trimesh = cast(Trimesh, trimesh.load(filepath))
        landmarks: NDArray = np.loadtxt(landmarks_filepath(filepath))
        color: str = COLORS[i % len(COLORS)]
        plotter.add_mesh(mesh=mesh, color=color, opacity=0.2)
        plotter.add_point_labels(
            points=landmarks,
            labels=range(landmarks.shape[0]),
            point_color=color,
            point_size=16,
            render_points_as_spheres=True,
            always_visible=True,
        )
    plotter.show()


if __name__ == "__main__":
    run(main)
