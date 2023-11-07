import dataclasses
import logging
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Annotated, cast

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


@dataclasses.dataclass(kw_only=True)
class Data:
    meshes: dict[Name, PolyData]
    landmarks: dict[Name, NDArray]
    colors: dict[Name, str]

    plotter: Plotter
    slider: vtkSliderWidget

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
        self.plotter.add_mesh(
            mesh=self.meshes[name], color=self.colors[name], opacity=0.3, name=name
        )

    def plot_landmarks(self, name: Name) -> None:
        mesh: PolyData = self.meshes[name]
        landmarks: NDArray = self.landmarks[name]
        if np.all(landmarks < 0):
            self.plotter.remove_actor(f"{name}-landmarks")  # type: ignore
        else:
            self.plotter.add_point_labels(
                points=mesh.points[landmarks[landmarks >= 0]],
                labels=np.arange(landmarks.shape[0])[landmarks >= 0],
                point_color=self.colors[name],
                point_size=16,
                name=f"{name}-landmarks",
                render_points_as_spheres=True,
                reset_camera=False,
                always_visible=True,
            )

    def plot_active_landmark(self, name: Name) -> None:
        mesh: PolyData = self.meshes[name]
        landmarks: NDArray = self.landmarks[name]
        if landmarks[self.active_index] < 0:
            self.plotter.remove_actor(f"{name}-landmark-active")  # type: ignore
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


def callback_pick(data: Data) -> Callable[[NDArray], None]:
    def callback(point: NDArray) -> None:
        data.pick(point=point)

    return callback


def callback_align(data: Data) -> Callable[[], None]:
    def callback() -> None:
        data.align()

    return callback


def callback_delete(data: Data) -> Callable[[], None]:
    def callback() -> None:
        data.delete()

    return callback


def callback_next(data: Data) -> Callable[[], None]:
    def callback() -> None:
        data.active_index += 1

    return callback


def callback_previous(data: Data) -> Callable[[], None]:
    def callback() -> None:
        data.active_index -= 1

    return callback


def callback_slider(data: Data) -> Callable[[float], None]:
    def callback(value: float) -> None:
        data.active_index = round(value)

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
    plotter: Plotter = Plotter()
    data: Data = Data(
        meshes={Name.SOURCE: source, Name.TARGET: target},
        landmarks={Name.SOURCE: source_landmarks, Name.TARGET: target_landmarks},
        colors={Name.SOURCE: "red", Name.TARGET: "green"},
        plotter=plotter,
        slider=vtkSliderWidget(),
    )
    plotter.enable_surface_point_picking(callback=callback_pick(data=data))
    plotter.add_key_event("a", callback_align(data=data))  # type: ignore
    plotter.add_key_event("d", callback_delete(data=data))  # type: ignore
    plotter.add_key_event("Right", callback_next(data=data))  # type: ignore
    plotter.add_key_event("Left", callback_previous(data=data))  # type: ignore
    slider: vtkSliderWidget = plotter.add_slider_widget(
        callback=callback_slider(data=data),
        rng=(0, source_landmarks.shape[0] - 1),
        title="Landmark Index",
        fmt="%.f",
    )
    data.slider = slider
    data.plot_mesh(name=Name.SOURCE)
    data.plot_mesh(name=Name.TARGET)
    data.plot_landmarks(name=Name.SOURCE)
    data.plot_landmarks(name=Name.TARGET)
    data.align()
    data.active_index = 0
    plotter.show()


if __name__ == "__main__":
    run(main)
