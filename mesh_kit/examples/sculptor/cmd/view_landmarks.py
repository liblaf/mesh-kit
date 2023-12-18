import pathlib
from collections.abc import Sequence
from typing import Annotated

import numpy as np
import trimesh
import typer
from numpy import typing as npt
from pyvista.plotting import plotter

from mesh_kit.common import cli, path

COLORS: Sequence[str] = ["green", "red"]


def main(
    mesh_filepath: Annotated[
        list[pathlib.Path], typer.Argument(exists=True, dir_okay=False)
    ],
) -> None:
    plot: plotter.Plotter = plotter.Plotter()
    for i, filepath in enumerate(mesh_filepath):
        mesh: trimesh.Trimesh = trimesh.load(filepath)
        landmarks: npt.NDArray = np.loadtxt(path.landmarks(filepath))
        color: str = COLORS[i % len(COLORS)]
        plot.add_mesh(mesh=mesh, color=color, opacity=0.2)
        plot.add_point_labels(
            points=landmarks,
            labels=range(landmarks.shape[0]),
            point_color=color,
            point_size=16,
            render_points_as_spheres=True,
            always_visible=True,
        )
    plot.show()


if __name__ == "__main__":
    cli.run(main)
    cli.run(main)
