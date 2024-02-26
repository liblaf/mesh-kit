import pathlib
from typing import Annotated

import numpy as np
import pyvista as pv
import trimesh
import typer
from numpy import typing as npt
from pyvista.plotting import plotter as _plotter
from pyvista.plotting import renderer as _renderer
from pyvista.plotting import themes as _themes

from mesh_kit import cli as _cli
from mesh_kit.io import trimesh as _io


def main(
    files: Annotated[list[pathlib.Path], typer.Argument(exists=True, dir_okay=False)],
) -> None:
    _themes.set_plot_theme("document_pro")
    plotter = _plotter.Plotter()
    opacity: float = 1.0 if len(files) == 1 else 0.5
    renderer: _renderer.Renderer = plotter.renderer
    for file in files:
        mesh: trimesh.Trimesh
        attrs: dict[str, npt.NDArray]
        mesh, attrs = _io.read(file, attr=True)
        color: str = renderer.next_color
        plotter.add_mesh(pv.wrap(mesh), color=color, opacity=opacity)
        points: npt.NDArray | None = None
        if "landmarks" in attrs:
            landmarks: npt.NDArray = attrs["landmarks"]
            points = mesh.vertices[landmarks]
        elif (
            landmarks_file := file.with_stem(file.stem + "-landmarks").with_suffix(
                ".txt"
            )
        ).exists():
            points = np.loadtxt(landmarks_file)
        else:
            points = None
        if points is not None:
            plotter.add_point_labels(
                points=points,
                labels=range(1, points.shape[0] + 1),
                font_size=32,
                point_color=color,
                point_size=32,
                render_points_as_spheres=True,
                always_visible=True,
            )
    plotter.show()


if __name__ == "__main__":
    _cli.run(main)
