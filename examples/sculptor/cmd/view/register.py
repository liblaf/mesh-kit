import dataclasses
import functools
import itertools
import json
import pathlib
from typing import Annotated, Any, Optional

import numpy as np
import pyvista as pv
import trimesh
import typer
import yaml
from loguru import logger
from numpy import typing as npt
from pyvista import plotting
from vtkmodules import vtkInteractionWidgets

from mesh_kit import convert
from mesh_kit.common import cli as _cli
from mesh_kit.pyvista import plotting as _plotting
from mesh_kit.pyvista.plotting import text as _text
from mesh_kit.registration import config as _config
from mesh_kit.registration import correspondence as _correspondence
from mesh_kit.registration import utils


@functools.cache
def read_source_mesh(directory: pathlib.Path, index: int) -> pv.PolyData:
    return pv.read(directory / f"{index:02d}.ply")


@functools.cache
def source_mesh_normalized(directory: pathlib.Path, index: int) -> trimesh.Trimesh:
    init_mesh: trimesh.Trimesh = convert.polydata2trimesh(
        read_source_mesh(directory, 0)
    )
    source_mesh: trimesh.Trimesh = convert.polydata2trimesh(
        read_source_mesh(directory, index)
    )
    return utils.normalize(
        source_mesh, centroid=init_mesh.centroid, scale=init_mesh.scale
    )


@functools.cache
def read_params(directory: pathlib.Path, index: int) -> Any | None:
    file: pathlib.Path = directory / f"{index:02d}-params.json"
    if file.exists():
        return json.loads(file.read_text())
    return None


@dataclasses.dataclass(kw_only=True)
class UI:
    directory: pathlib.Path
    plotter: plotting.Plotter = dataclasses.field(default_factory=plotting.Plotter)
    _cache_correspondence: dict[
        int, tuple[npt.NDArray, npt.NDArray]
    ] = dataclasses.field(default_factory=dict)
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

        def callback(x: float) -> None:
            self.index = x

        self.plotter.add_slider_widget(callback=callback, rng=(0, self.num_records - 1))

        def callback() -> None:
            logger.debug("Key Pressed: 'j'")
            self.index += 1

        self.plotter.add_key_event("j", callback)

        def callback() -> None:
            logger.debug("Key Pressed: 'k'")
            self.index -= 1

        self.plotter.add_key_event("k", callback)

        def callback() -> None:
            logger.debug("Key Pressed: 'o'")
            self.opacity = next(self._opacity_cycle)

        self.plotter.add_key_event("o", callback)

    def show(self) -> None:
        self.index = 0
        self.opacity = next(self._opacity_cycle)
        self.plot_target_landmarks()
        self.plotter.show()

    def plot_source_mesh(self) -> None:
        self.plotter.add_mesh(
            self.source_mesh(index=self.index),
            color=self.source_color,
            opacity=self.opacity,
            name="source-mesh",
        )

    def plot_target_mesh(self) -> None:
        self.plotter.add_mesh(
            self.target_mesh,
            color=self.target_color,
            opacity=self.opacity,
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
            font_size=_plotting.POINT_SIZE,
            point_color=self.source_color,
            point_size=_plotting.POINT_SIZE,
            name="source-landmarks",
            render_points_as_spheres=True,
            always_visible=True,
        )

    def plot_target_landmarks(self) -> None:
        num_landmarks: int = self.target_positions.shape[0]
        self.plotter.add_point_labels(
            points=self.target_positions,
            labels=[str(i) for i in range(1, num_landmarks + 1)],
            font_size=_plotting.POINT_SIZE,
            point_color=self.target_color,
            point_size=_plotting.POINT_SIZE,
            name="target-landmarks",
            render_points_as_spheres=True,
            always_visible=True,
        )

    def plot_params(self) -> None:
        params: Optional[_config.Params] = self.params(self.index)
        text: str = yaml.dump(params.model_dump() if params else None)
        self.plotter.add_text(text, name="params", font_file=_text.monospace())

    def plot_correspondence(self) -> None:
        source_positions: npt.NDArray
        target_positions: npt.NDArray
        source_positions, target_positions = self.correspondence(self.index)
        self.plotter.add_arrows(
            cent=source_positions,
            direction=target_positions - source_positions,
            name="correspondence",
        )

    @functools.cached_property
    def num_records(self) -> pv.PolyData:
        for i in itertools.count(0):
            if not (self.directory / f"{i:02d}.ply").is_file():
                return i
        return None

    def source_mesh(self, index: int) -> pv.PolyData:
        return read_source_mesh(self.directory, index)

    def source_mesh_normalized(self, index: int) -> trimesh.Trimesh:
        return source_mesh_normalized(self.directory, index)

    @functools.cached_property
    def target_mesh(self) -> pv.PolyData:
        return pv.read(self.directory / "target.ply")

    @functools.cached_property
    def target_mesh_normalized(self) -> trimesh.Trimesh:
        source_mesh: trimesh.Trimesh = convert.polydata2trimesh(self.source_mesh(0))
        target_mesh: trimesh.Trimesh = convert.polydata2trimesh(self.target_mesh)
        return utils.normalize(
            target_mesh, centroid=source_mesh.centroid, scale=source_mesh.scale
        )

    @functools.cached_property
    def source_landmarks(self) -> npt.NDArray:
        return np.loadtxt(self.directory / "source-landmarks.txt", dtype=int)

    @functools.cached_property
    def target_positions(self) -> npt.NDArray:
        return np.loadtxt(self.directory / "target-positions.txt")

    def params(self, index: int) -> Optional[_config.Params]:
        data: Any = read_params(self.directory, index)
        if data is None:
            return None
        return _config.Params(**data)

    def correspondence(self, index: int) -> npt.NDArray:
        if index not in self._cache_correspondence:
            origin_source_mesh: trimesh.Trimesh = convert.polydata2trimesh(
                self.source_mesh(0)
            )
            denormalize = functools.partial(
                utils.denormalize,
                centroid=origin_source_mesh.centroid,
                scale=origin_source_mesh.scale,
            )
            source_mesh: trimesh.Trimesh = self.source_mesh_normalized(index)
            target_mesh: trimesh.Trimesh = self.target_mesh_normalized
            distances: npt.NDArray
            target_positions: npt.NDArray
            if index + 1 < self.num_records:
                params: _config.Params = self.params(index)
            else:
                params: _config.Params = self.params(index - 1)
            distances, target_positions, _ = _correspondence.correspondence(
                source_mesh=source_mesh,
                target_mesh=target_mesh,
                config=params.correspondence,
            )
            source_positions: npt.NDArray = source_mesh.vertices
            valid: npt.NDArray = distances < params.correspondence.threshold
            source_positions = source_positions[valid]
            target_positions = target_positions[valid]
            source_positions = denormalize(source_positions)
            target_positions = denormalize(target_positions)
            self._cache_correspondence[index] = source_positions, target_positions
        return self._cache_correspondence[index]

    @property
    def slider(self) -> vtkInteractionWidgets.vtkSliderWidget:
        return self.plotter.slider_widgets[0]

    @property
    def index(self) -> int:
        return round(self.slider.GetSliderRepresentation().GetValue())

    @index.setter
    def index(self, value: int) -> None:
        self.slider.GetSliderRepresentation().SetValue(
            round(np.clip(value, 0, self.num_records - 1))
        )
        self.plot_source_mesh()
        self.plot_source_landmarks()
        self.plot_params()
        self.plot_correspondence()

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
