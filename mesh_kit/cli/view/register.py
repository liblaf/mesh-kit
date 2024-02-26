import dataclasses
import functools
import itertools
import json
import pathlib
from typing import Annotated

import pyvista as pv
import trimesh
import typer
import yaml
from loguru import logger
from numpy import typing as npt
from pyvista import plotting
from vtkmodules import vtkInteractionWidgets

from mesh_kit import cli as _cli
from mesh_kit import polydata as _poly
from mesh_kit import tetgen as _tetgen
from mesh_kit.io import trimesh as _io
from mesh_kit.plot import font as _font
from mesh_kit.register import config as _config
from mesh_kit.register import nearest as _nearest


@functools.cache
def read_init_mesh(
    directory: pathlib.Path,
) -> tuple[trimesh.Trimesh, dict[str, npt.NDArray]]:
    return _io.read(directory / "source.ply", attr=True)


@functools.cache
def read_target_mesh(
    directory: pathlib.Path,
) -> tuple[trimesh.Trimesh, dict[str, npt.NDArray]]:
    return _io.read(directory / "target.ply", attr=True)


@functools.cache
def read_source_mesh(
    directory: pathlib.Path, index: int
) -> tuple[trimesh.Trimesh, dict[str, npt.NDArray]]:
    mesh: trimesh.Trimesh = _io.read(directory / f"{index:02d}.ply", attr=False)
    init_mesh: trimesh.Trimesh
    init_attrs: dict[str, npt.NDArray]
    init_mesh, init_attrs = read_init_mesh(directory)
    return mesh, init_attrs


@functools.cache
def read_params(directory: pathlib.Path, index: int) -> _config.Params:
    file: pathlib.Path = directory / f"{index:02d}-params.json"
    return _config.Params(**json.loads(file.read_text()))


@functools.cache
def nearest(directory: pathlib.Path, index: int) -> tuple[npt.NDArray, npt.NDArray]:
    source_mesh: trimesh.Trimesh
    source_attrs: dict[str, npt.NDArray]
    source_mesh, source_attrs = read_source_mesh(directory, index)
    target_mesh: trimesh.Trimesh
    target_attrs: dict[str, npt.NDArray]
    target_mesh, target_attrs = read_target_mesh(directory)
    params: _config.Params = read_params(directory, index)
    distance: npt.NDArray
    target_positions: npt.NDArray
    target_normals: npt.NDArray
    distance, target_positions, target_normals = _nearest.nearest(
        source_mesh=source_mesh, target_mesh=target_mesh, config=params.nearest
    )
    threshold: float | npt.NDArray = source_attrs.get(
        "vert:distance-threshold", params.nearest.threshold
    )
    mask: npt.NDArray = distance < threshold
    return source_mesh.vertices[mask], target_positions[mask]


@functools.cache
def check_watertight(directory: pathlib.Path, index: int) -> bool:
    mesh: trimesh.Trimesh
    attrs: dict[str, npt.NDArray]
    mesh, attrs = read_source_mesh(directory, index)
    return _tetgen.check(mesh)


@dataclasses.dataclass(kw_only=True)
class UI:
    directory: pathlib.Path
    plotter: plotting.Plotter = dataclasses.field(default_factory=plotting.Plotter)
    _opacity: float = 1.0
    _opacity_cycle: itertools.cycle = dataclasses.field(
        default_factory=lambda: itertools.cycle([0.2, 0.5, 1.0])
    )
    source_color: plotting.ColorLike = dataclasses.field(init=False)
    target_color: plotting.ColorLike = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        render: plotting.renderer.Renderer = self.plotter.renderer
        self.source_color = render.next_color
        self.target_color = render.next_color

        def callback_index(x: float) -> None:
            self.index = round(x)

        self.plotter.add_slider_widget(
            callback=callback_index, rng=(0, self.num_records - 1)
        )

        def callback_next() -> None:
            logger.debug("Key Pressed: 'j'")
            self.index += 1

        self.plotter.add_key_event("j", callback_next)

        def callback_prev() -> None:
            logger.debug("Key Pressed: 'k'")
            self.index -= 1

        self.plotter.add_key_event("k", callback_prev)

        def callback_opacity() -> None:
            logger.debug("Key Pressed: 'o'")
            self.opacity = next(self._opacity_cycle)

        self.plotter.add_key_event("o", callback_opacity)

    def show(self) -> None:
        self.index = 0
        self.opacity = next(self._opacity_cycle)
        self.plot_target_landmarks()
        self.plotter.add_legend(bcolor=None)  # pyright: ignore
        self.plotter.show()

    def plot_source_mesh(self) -> None:
        self.plotter.add_mesh(
            self.source_mesh(index=self.index),
            color=self.source_color,
            opacity=self.opacity,
            label="Source",
            name="source-mesh",
        )

    def plot_target_mesh(self) -> None:
        self.plotter.add_mesh(
            self.target_mesh,
            color=self.target_color,
            opacity=self.opacity,
            label="Target",
            name="target-mesh",
        )

    def plot_source_landmarks(self) -> None:
        source_positions: npt.NDArray = self.source_mesh(self.index).points[
            self.source_landmarks
        ]
        num_landmarks: int = self.source_landmarks.shape[0]
        self.plotter.add_point_labels(
            points=source_positions,
            labels=[str(i) for i in range(1, num_landmarks + 1)],
            font_size=32,
            point_color=self.source_color,
            point_size=32,
            name="source-landmarks",
            render_points_as_spheres=True,
            always_visible=True,
        )

    def plot_target_landmarks(self) -> None:
        num_landmarks: int = self.target_positions.shape[0]
        self.plotter.add_point_labels(
            points=self.target_positions,
            labels=[str(i) for i in range(1, num_landmarks + 1)],
            font_size=32,
            point_color=self.target_color,
            point_size=32,
            name="target-landmarks",
            render_points_as_spheres=True,
            always_visible=True,
        )

    def plot_params(self) -> None:
        params: _config.Params = self.params(self.index)
        text: str = yaml.dump(params.model_dump(), indent=2, sort_keys=False)
        self.plotter.add_text(text, name="params", font_file=_font.monospace())

    def plot_nearest(self) -> None:
        source_positions: npt.NDArray
        target_positions: npt.NDArray
        source_positions, target_positions = self.nearest(self.index)
        self.plotter.add_arrows(
            cent=source_positions,
            direction=target_positions - source_positions,
            name="nearest",
        )

    def plot_info(self) -> None:
        watertight: bool = check_watertight(self.directory, self.index)
        text: str = f"Watertight: {watertight}"
        self.plotter.add_text(
            text,
            position="lower_left",
            color="green" if watertight else "red",
            name="info",
            font_file=_font.monospace(),
        )

    @functools.cached_property
    def num_records(self) -> int:
        for i in itertools.count(0):
            if not (self.directory / f"{i:02d}.ply").is_file():
                return i
        raise RuntimeError

    def source_mesh(self, index: int) -> pv.PolyData:
        mesh: trimesh.Trimesh
        attrs: dict[str, npt.NDArray]
        mesh, attrs = read_source_mesh(self.directory, index)
        return _poly.as_polydata(mesh)

    @functools.cached_property
    def target_mesh(self) -> pv.PolyData:
        mesh: trimesh.Trimesh
        attrs: dict[str, npt.NDArray]
        mesh, attrs = read_target_mesh(self.directory)
        return _poly.as_polydata(mesh)

    @functools.cached_property
    def source_landmarks(self) -> npt.NDArray:
        mesh: trimesh.Trimesh
        attrs: dict[str, npt.NDArray]
        mesh, attrs = read_source_mesh(self.directory, 0)
        return attrs["landmarks"]

    @functools.cached_property
    def target_positions(self) -> npt.NDArray:
        mesh: trimesh.Trimesh
        attrs: dict[str, npt.NDArray]
        mesh, attrs = read_target_mesh(self.directory)
        return mesh.vertices[attrs["landmarks"]]

    def params(self, index: int) -> _config.Params:
        return read_params(self.directory, index)

    def nearest(self, index: int) -> tuple[npt.NDArray, npt.NDArray]:
        return nearest(self.directory, index)

    @property
    def slider(self) -> vtkInteractionWidgets.vtkSliderWidget:
        return self.plotter.slider_widgets[0]

    @property
    def index(self) -> int:
        return round(self.slider.GetSliderRepresentation().GetValue())

    @index.setter
    def index(self, value: float) -> None:
        self.slider.GetSliderRepresentation().SetValue(
            round(min(max(value, 0), self.num_records - 1))
        )
        self.plot_source_mesh()
        self.plot_source_landmarks()
        self.plot_params()
        self.plot_nearest()
        self.plot_info()

    @property
    def opacity(self) -> float:
        return self._opacity

    @opacity.setter
    def opacity(self, opacity: float) -> None:
        self._opacity = opacity
        self.plot_source_mesh()
        self.plot_target_mesh()


def main(
    directory: Annotated[pathlib.Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    plotting.set_plot_theme("document_pro")
    ui = UI(directory=directory)
    ui.show()


if __name__ == "__main__":
    _cli.run(main)
