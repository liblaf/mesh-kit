import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from pyvista import PolyData
from pyvista.plotting.plotter import Plotter
from typer import Argument

from mesh_kit.common.cli import run


def slider_callback(
    plotter: Plotter,
    records: Sequence[PolyData],
    source_landmarks: Sequence[NDArray],
    target_landmarks: Sequence[NDArray],
) -> Callable[[float], None]:
    def callback(value: float) -> None:
        t: int = math.floor(value)
        plotter.add_mesh(
            mesh=records[t], color="red", opacity=0.2, name="source", pickable=False
        )
        plotter.add_point_labels(
            points=source_landmarks[t],
            labels=[""] * len(source_landmarks[t]),
            point_color="red",
            point_size=16,
            name="source-landmarks",
            render_points_as_spheres=True,
            always_visible=True,
        )
        plotter.add_point_labels(
            points=target_landmarks[t],
            labels=[""] * len(target_landmarks[t]),
            point_color="green",
            point_size=16,
            name="target-landmarks",
            render_points_as_spheres=True,
            always_visible=True,
        )

    return callback


def main(
    records_dirpath: Annotated[Path, Argument(exists=True, file_okay=False)],
    target_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
) -> None:
    records_filepath: Sequence[Path] = sorted(list(records_dirpath.glob("*.ply")))
    records: Sequence[PolyData] = [
        cast(PolyData, pv.read(filepath)) for filepath in records_filepath
    ]
    source_landmarks: Sequence[NDArray] = [
        np.loadtxt(
            filepath.with_stem(filepath.stem + "-source-landmarks").with_suffix(".txt")
        )
        for filepath in records_filepath
    ]
    target: PolyData = cast(PolyData, pv.read(target_filepath))
    target_landmarks: Sequence[NDArray] = [
        np.loadtxt(
            filepath.with_stem(filepath.stem + "-target-landmarks").with_suffix(".txt")
        )
        for filepath in records_filepath
    ]
    plotter: Plotter = Plotter()

    plotter.add_slider_widget(
        callback=slider_callback(
            plotter=plotter,
            records=records,
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
        ),
        rng=(0, len(records)),
        value=0,
    )
    plotter.add_mesh(mesh=target, color="green", opacity=0.2)
    plotter.add_floor()

    def callback(point, picker) -> None:
        print(type(point), point)
        print(type(picker), picker)

    plotter.enable_surface_point_picking(
        callback=print, color="blue", point_size=16, render_points_as_spheres=True
    )
    plotter.show()


if __name__ == "__main__":
    run(main)
