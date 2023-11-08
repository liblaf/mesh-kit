import dataclasses
import logging
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional, cast

import numpy as np
import pyvista as pv
import trimesh.registration
from numpy.typing import NDArray
from pyvista import PolyData
from pyvista.plotting.plotter import Plotter
from typer import Argument
from vtkmodules.vtkInteractionWidgets import vtkSliderWidget

import mesh_kit.registration.landmarks
from mesh_kit.common.cli import run
from mesh_kit.common.path import landmarks_filepath


class Name(str, Enum):
    SOURCE = "source"
    TARGET = "target"


class UI:
    meshes: dict[Name, PolyData]
    landmarks: dict[Name, NDArray]

    plotter: Plotter
    colors: dict[Name, str]
    visible: dict[Name, bool]

    def __init__(
        self,
        source: PolyData,
        source_landmarks: NDArray,
        target: PolyData,
        target_landmarks: NDArray,
    ) -> None:
        self.meshes = {Name.SOURCE: source, Name.TARGET: target}
        self.landmarks = {Name.SOURCE: source_landmarks, Name.TARGET: target_landmarks}

        self.plotter = Plotter()
        self.colors = {Name.SOURCE: "red", Name.TARGET: "green"}
        self.visible = {Name.SOURCE: True, Name.TARGET: True}

        self.plotter.enable_surface_point_picking(callback=callback_pick(ui=self))
        self.plotter.add_key_event("a", callback_align(ui=self))  # type: ignore
        self.plotter.add_key_event("d", callback_delete(ui=self))  # type: ignore
        self.plotter.add_key_event("h", callback_hide(ui=self))  # type: ignore
        self.plotter.add_key_event("Right", callback_next(ui=self))  # type: ignore
        self.plotter.add_key_event("Left", callback_previous(ui=self))  # type: ignore
        self.plotter.add_slider_widget(
            callback=callback_slider(ui=self),
            rng=(0, self.landmarks[Name.SOURCE].shape[0] - 1),
            title="Landmark Index",
            fmt="%.f",
        )
        self.align()
        self.active_index = 0
        self.plot_mesh(name=Name.TARGET)
        self.plot_landmarks(name=Name.TARGET)

    def align(self) -> None:
        matrix: NDArray
        cost: float
        valid_landmarks: NDArray = (self.source_landmarks >= 0) & (
            self.target_landmarks >= 0
        )
        if np.count_nonzero(valid_landmarks) < 3:
            logging.error("Not enough landmarks to align")
            return
        matrix, _, cost = trimesh.registration.procrustes(
            self.source.points[self.source_landmarks[valid_landmarks]],
            self.target.points[self.target_landmarks[valid_landmarks]],
        )
        logging.info(f"Rigid Align Cost: {cost}")
        self.source.transform(matrix, inplace=True)
        self.plot_mesh(name=Name.SOURCE)
        self.plot_landmarks(name=Name.SOURCE)
        self.plot_active_landmark(name=Name.SOURCE)

    def delete(self) -> None:
        self.target_landmarks[self.active_index] = -1
        self.plot_landmarks(name=Name.TARGET)
        self.plot_active_landmark(name=Name.TARGET)

    def pick(self, point: NDArray) -> None:
        logging.info(f"Pick {self.active_index}: {point}")
        self.target_landmarks[
            self.active_index
        ] = mesh_kit.registration.landmarks.position_to_index(
            mesh=self.target, position=point
        )
        self.plot_landmarks(name=Name.TARGET)
        self.plot_active_landmark(name=Name.TARGET)

    def plot(self) -> None:
        self.plot_mesh(name=Name.SOURCE)
        self.plot_mesh(name=Name.TARGET)
        self.plot_landmarks(name=Name.SOURCE)
        self.plot_landmarks(name=Name.TARGET)
        self.plot_active_landmark(name=Name.SOURCE)
        self.plot_active_landmark(name=Name.TARGET)

    def plot_mesh(self, name: Name) -> None:
        if self.visible[name]:
            color: Optional[str] = (
                self.colors[name] if self.visible[Name.SOURCE] else None
            )
            opacity: Optional[float] = 0.3 if self.visible[Name.SOURCE] else None
            self.plotter.add_mesh(
                mesh=self.meshes[name],
                color=color,
                opacity=opacity,
                reset_camera=False,
                name=f"{name}-mesh",
                smooth_shading=True,
                pickable=(name == Name.TARGET),
            )
        else:
            self.plotter.remove_actor(actor=f"{name}-mesh", reset_camera=False)  # type: ignore

    def plot_landmarks(self, name: Name) -> None:
        mesh: PolyData = self.meshes[name]
        landmarks: NDArray = self.landmarks[name]
        if np.all(landmarks < 0):
            self.plotter.remove_actor(actor=f"{name}-landmarks", reset_camera=False)  # type: ignore
        else:
            self.plotter.add_point_labels(
                points=mesh.points[landmarks[landmarks >= 0]],
                labels=np.arange(landmarks.shape[0])[landmarks >= 0],
                point_color=self.colors[name],
                point_size=16,
                name=f"{name}-landmarks",
                shape_color=self.colors[name],
                render_points_as_spheres=True,
                reset_camera=False,
                always_visible=True,
            )

    def plot_active_landmark(self, name: Name) -> None:
        mesh: PolyData = self.meshes[name]
        landmarks: NDArray = self.landmarks[name]
        if landmarks[self.active_index] < 0:
            self.plotter.remove_actor(
                actor=f"{name}-landmark-active", reset_camera=False
            )  # type: ignore
        else:
            self.plotter.add_points(
                points=mesh.points[landmarks[self.active_index]],
                color=self.colors[name],
                point_size=32,
                reset_camera=False,
                name=f"{name}-landmark-active",
                render_points_as_spheres=True,
            )

    @property
    def source(self) -> PolyData:
        return self.meshes[Name.SOURCE]

    @property
    def target(self) -> PolyData:
        return self.meshes[Name.TARGET]

    @property
    def source_landmarks(self) -> NDArray:
        return self.landmarks[Name.SOURCE]

    @property
    def target_landmarks(self) -> NDArray:
        return self.landmarks[Name.TARGET]

    @property
    def active_index(self) -> int:
        return round(self.slider.GetSliderRepresentation().GetValue())

    @active_index.setter
    def active_index(self, index: int) -> None:
        index = np.clip(index, 0, self.source_landmarks.shape[0] - 1)
        self.slider.GetSliderRepresentation().SetValue(index)
        self.plot_active_landmark(name=Name.SOURCE)
        self.plot_active_landmark(name=Name.TARGET)

    @property
    def slider(self) -> vtkSliderWidget:
        return self.plotter.slider_widgets[0]


def callback_pick(ui: UI) -> Callable[[NDArray], None]:
    def callback(point: NDArray) -> None:
        ui.pick(point=point)

    return callback


def callback_align(ui: UI) -> Callable[[], None]:
    def callback() -> None:
        ui.align()

    return callback


def callback_delete(ui: UI) -> Callable[[], None]:
    def callback() -> None:
        ui.delete()

    return callback


def callback_hide(ui: UI) -> Callable[[], None]:
    def callback() -> None:
        ui.visible[Name.SOURCE] = not ui.visible[Name.SOURCE]
        ui.plot_mesh(name=Name.SOURCE)
        ui.plot_mesh(name=Name.TARGET)

    return callback


def callback_next(ui: UI) -> Callable[[], None]:
    def callback() -> None:
        ui.active_index += 1

    return callback


def callback_previous(ui: UI) -> Callable[[], None]:
    def callback() -> None:
        ui.active_index -= 1

    return callback


def callback_slider(ui: UI) -> Callable[[float], None]:
    def callback(value: float) -> None:
        ui.active_index = round(value)

    return callback


def main(
    source_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    target_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
) -> None:
    source: PolyData = cast(PolyData, pv.read(source_filepath))
    target: PolyData = cast(PolyData, pv.read(target_filepath))
    source_landmarks: NDArray = mesh_kit.registration.landmarks.position_to_index(
        mesh=source, position=np.loadtxt(landmarks_filepath(source_filepath))
    )
    target_landmarks: NDArray = mesh_kit.registration.landmarks.position_to_index(
        mesh=target, position=np.loadtxt(landmarks_filepath(target_filepath))
    )
    target_landmarks = np.resize(target_landmarks, new_shape=source_landmarks.shape)
    ui: UI = UI(
        source=source,
        source_landmarks=source_landmarks,
        target=target,
        target_landmarks=target_landmarks,
    )
    ui.plotter.show()


if __name__ == "__main__":
    run(main)
