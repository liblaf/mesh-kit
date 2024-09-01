import pathlib

import confz
import pyvista as pv
from confz import BaseConfig
from icecream import ic

from mkit.typing import StrPath


class Camera(BaseConfig):
    focal_point: list[float]
    position: list[float]
    up: list[float]
    window_size: list[int]
    view_angle: float
    parallel_projection: bool
    parallel_scale: float


def load_camera(pl: pv.Plotter, file: StrPath) -> None:
    data = Camera(config_sources=[confz.FileSource(file)])
    camera: pv.Camera = pl.camera
    camera.focal_point = data.focal_point
    camera.parallel_projection = data.parallel_projection
    camera.parallel_scale = data.parallel_scale
    camera.position = data.position
    camera.up = data.up
    camera.view_angle = data.view_angle
    pl.window_size = data.window_size


def save_camera(_file: StrPath, pl: pv.Plotter) -> None:
    file: pathlib.Path = pathlib.Path(_file)
    camera: pv.Camera = pl.camera
    data = Camera(
        focal_point=camera.focal_point,
        parallel_projection=camera.parallel_projection,
        parallel_scale=camera.parallel_scale,
        position=camera.position,
        up=camera.up,
        view_angle=camera.view_angle,
        window_size=pl.window_size,
    )
    ic(data)
    file.write_text(data.model_dump_json())
