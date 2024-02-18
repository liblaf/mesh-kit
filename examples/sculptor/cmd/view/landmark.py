import pathlib
from typing import TYPE_CHECKING, Annotated

import pyvista as pv
import typer
from pyvista.plotting import plotter as _plotter
from pyvista.plotting import renderer as _renderer
from pyvista.plotting import themes as _themes

from mesh_kit.common import cli as _cli
from mesh_kit.io import landmark as _landmark

if TYPE_CHECKING:
    from numpy import typing as npt


def main(
    files: Annotated[list[pathlib.Path], typer.Argument(exists=True, dir_okay=False)],
) -> None:
    _themes.set_plot_theme("document_pro")
    plotter = _plotter.Plotter()
    opacity: float = 1.0 if len(files) == 1 else 0.5
    renderer: _renderer.Renderer = plotter.renderer
    for file in files:
        mesh: pv.PolyData = pv.read(file)
        landmarks: npt.NDArray = _landmark.read(file)
        color: str = renderer.next_color
        point_size: int = max(plotter.window_size) // 16
        plotter.add_mesh(mesh, color=color, opacity=opacity)
        plotter.add_point_labels(
            points=landmarks,
            labels=range(1, len(landmarks) + 1),
            font_size=point_size,
            point_color=color,
            point_size=point_size,
            render_points_as_spheres=True,
            always_visible=True,
        )
    plotter.show()


if __name__ == "__main__":
    _cli.run(main)
