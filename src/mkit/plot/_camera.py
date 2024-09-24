import pydantic
import pyvista as pv

from mkit import utils
from mkit.typing import StrPath


class CameraParams(pydantic.BaseModel):
    focal_point: list[float]
    position: list[float]
    up: list[float]
    window_size: list[int]
    view_angle: float
    parallel_projection: bool
    parallel_scale: float


def load_camera(pl: pv.Plotter, fpath: StrPath) -> None:
    data: CameraParams = utils.load_pydantic(CameraParams, fpath)
    camera: pv.Camera = pl.camera
    camera.focal_point = data.focal_point
    camera.parallel_projection = data.parallel_projection
    camera.parallel_scale = data.parallel_scale
    camera.position = data.position
    camera.up = data.up
    camera.view_angle = data.view_angle
    pl.window_size = data.window_size


def save_camera(pl: pv.Plotter, fpath: StrPath) -> None:
    camera: pv.Camera = pl.camera
    data = CameraParams(
        focal_point=camera.focal_point,  # pyright: ignore [reportArgumentType]
        parallel_projection=camera.parallel_projection,
        parallel_scale=camera.parallel_scale,
        position=camera.position,  # pyright: ignore [reportArgumentType]
        up=camera.up,  # pyright: ignore [reportArgumentType]
        view_angle=camera.view_angle,
        window_size=pl.window_size,
    )
    utils.save_pydantic(data, fpath)
