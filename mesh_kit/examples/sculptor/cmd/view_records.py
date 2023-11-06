from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from pyvista import PolyData
from pyvista.plotting.plotter import Plotter
from typer import Argument
from vtkmodules.vtkInteractionWidgets import vtkSliderWidget

from mesh_kit.common.cli import run


def slider_callback(
    plotter: Plotter,
    records: Sequence[PolyData],
    source_landmarks: Sequence[NDArray],
    target_landmarks: Sequence[NDArray],
) -> Callable[[float, vtkSliderWidget], None]:
    def callback(value: float, widget: vtkSliderWidget) -> None:
        widget.GetSliderRepresentation().SetValue(round(value))
        t: int = round(widget.GetSliderRepresentation().GetValue())
        plotter.add_mesh(
            mesh=records[t], color="red", opacity=0.2, name="source", pickable=False
        )
        plotter.add_points(
            points=source_landmarks[t],
            color="red",
            point_size=16,
            name="source-landmarks",
            render_points_as_spheres=True,
        )
        plotter.add_points(
            points=target_landmarks[t],
            color="green",
            point_size=16,
            name="target-landmarks",
            render_points_as_spheres=True,
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
    callback: Callable[[float, vtkSliderWidget], None] = slider_callback(
        plotter=plotter,
        records=records,
        source_landmarks=source_landmarks,
        target_landmarks=target_landmarks,
    )
    slider: vtkSliderWidget = plotter.add_slider_widget(
        callback=callback,
        rng=(0, len(records) - 1),
        value=0,
        title="Step",
        pass_widget=True,
        fmt="%.f",
    )
    plotter.add_key_event(
        "n", lambda: callback(slider.GetSliderRepresentation().GetValue() + 1, slider)  # type: ignore
    )
    plotter.add_key_event(
        "p", lambda: callback(slider.GetSliderRepresentation().GetValue() - 1, slider)  # type: ignore
    )
    plotter.add_mesh(mesh=target, color="green", opacity=0.2)
    plotter.show()


if __name__ == "__main__":
    run(main)
