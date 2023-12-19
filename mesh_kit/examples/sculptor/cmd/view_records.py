import dataclasses
import pathlib
from collections.abc import Callable, Sequence
from typing import Annotated

import numpy as np
import pyvista as pv
import typer
from numpy import typing as npt
from pyvista.plotting.plotter import Plotter
from vtkmodules import vtkInteractionWidgets

from mesh_kit.common import cli


@dataclasses.dataclass(kw_only=True)
class UI:
    params: Sequence[str]
    source: Sequence[pv.PolyData]
    target: pv.PolyData
    source_landmarks: Sequence[npt.NDArray]
    target_landmarks: Sequence[npt.NDArray]

    plotter: Plotter

    def plot_params(self) -> None:
        self.plotter.add_text(text=self.params[self.step], name="params")

    def plot_source(self) -> None:
        self.plotter.add_mesh(
            mesh=self.source[self.step], color="red", opacity=0.2, name="source"
        )

    def plot_source_landmarks(self) -> None:
        if self.source_landmarks[self.step].size == 0:
            self.plotter.remove_actor("source-correspondence")
        else:
            self.plotter.add_points(
                points=self.source_landmarks[self.step],
                color="red",
                point_size=8,
                name="source-correspondence",
                render_points_as_spheres=True,
            )

    def plot_target(self) -> None:
        self.plotter.add_mesh(
            mesh=self.target, color="green", opacity=0.2, name="target"
        )

    def plot_correspondence(self) -> None:
        if self.source_landmarks[self.step].size == 0:
            self.plotter.remove_actor("correspondence")
        else:
            self.plotter.add_arrows(
                cent=self.source_landmarks[self.step],
                direction=self.target_landmarks[self.step]
                - self.source_landmarks[self.step],
                name="correspondence",
            )

    def plot_target_landmarks(self) -> None:
        if self.target_landmarks[self.step].size == 0:
            self.plotter.remove_actor("target-correspondence")
        else:
            self.plotter.add_points(
                points=self.target_landmarks[self.step],
                color="green",
                point_size=8,
                name="target-correspondence",
                render_points_as_spheres=True,
            )

    @property
    def step(self) -> int:
        return round(self.slider.GetSliderRepresentation().GetValue())

    @step.setter
    def step(self, value: int) -> None:
        self.slider.GetSliderRepresentation().SetValue(round(value))
        self.plot_params()
        self.plot_source()
        self.plot_correspondence()

    @property
    def slider(self) -> vtkInteractionWidgets.vtkSliderWidget:
        return self.plotter.slider_widgets[0]


def callback_slider(
    ui: UI,
) -> Callable[[float], None]:
    def callback(value: float) -> None:
        ui.step = round(value)

    return callback


def callback_next(ui: UI) -> Callable[[], None]:
    def callback() -> None:
        ui.step += 1

    return callback


def callback_previous(ui: UI) -> Callable[[], None]:
    def callback() -> None:
        ui.step -= 1

    return callback


def main(
    records_dirpath: Annotated[
        pathlib.Path, typer.Argument(exists=True, file_okay=False)
    ],
    target_filepath: Annotated[
        pathlib.Path, typer.Argument(exists=True, dir_okay=False)
    ],
) -> None:
    records_filepath: Sequence[pathlib.Path] = sorted(
        list(records_dirpath.glob("*.ply"))
    )
    records: Sequence[pv.PolyData] = [
        pv.read(filepath) for filepath in records_filepath
    ]
    params: Sequence[str] = [
        filepath.with_stem(filepath.stem + "-params").with_suffix(".json").read_text()
        for filepath in records_filepath
    ]
    source_landmarks: Sequence[npt.NDArray] = [
        np.loadtxt(
            filepath.with_stem(filepath.stem + "-source-correspondence").with_suffix(
                ".txt"
            )
        )
        for filepath in records_filepath
    ]
    target: pv.PolyData = pv.read(target_filepath)
    target_landmarks: Sequence[npt.NDArray] = [
        np.loadtxt(
            filepath.with_stem(filepath.stem + "-target-correspondence").with_suffix(
                ".txt"
            )
        )
        for filepath in records_filepath
    ]
    plotter: Plotter = Plotter()
    ui: UI = UI(
        params=params,
        source=records,
        target=target,
        source_landmarks=source_landmarks,
        target_landmarks=target_landmarks,
        plotter=plotter,
    )
    plotter.add_key_event("Right", callback_next(ui=ui))  # type: ignore
    plotter.add_key_event("Left", callback_previous(ui=ui))  # type: ignore
    plotter.add_slider_widget(
        callback=callback_slider(ui=ui),
        rng=(0, len(records) - 1),
        title="Step",
        fmt="%.f",
    )
    ui.plot_target()
    ui.step = 0
    plotter.show()


if __name__ == "__main__":
    cli.run(main)
