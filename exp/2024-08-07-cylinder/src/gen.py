import pathlib
from typing import Literal

import numpy as np
import numpy.typing as npt
import pyvista as pv

import mkit.cli
import mkit.ext
import mkit.transfer.point_data


class Config(mkit.cli.BaseConfig):
    deform: Literal["squash", "stretch", "twist"]
    output: pathlib.Path


def main(cfg: Config) -> None:
    surface: pv.PolyData
    match cfg.deform:
        case "squash":
            surface = pv.Cylinder(height=2)
        case "stretch":
            surface = pv.Cylinder()
        case "twist":
            surface = pv.Box((-2, 2, -1, 1, -1, 1))
    surface.triangulate(inplace=True, progress_bar=True)
    tetmesh: pv.UnstructuredGrid = mkit.ext.tetwild(surface)
    surface = tetmesh.extract_surface(progress_bar=True)
    left_mask: npt.NDArray[np.bool] = surface.points[:, 0] < surface.bounds[0] + 1e-3
    right_mask: npt.NDArray[np.bool] = surface.points[:, 0] > surface.bounds[1] - 1e-3
    surface.point_data["pin_mask"] = left_mask | right_mask
    pin_disp: npt.NDArray[np.floating] = np.zeros((surface.n_points, 3))
    match cfg.deform:
        case "squash":
            pin_disp[left_mask] = [0.5, 0, 0]
            pin_disp[right_mask] = [-0.5, 0, 0]
        case "stretch":
            pin_disp[left_mask] = [-0.5, 0, 0]
            pin_disp[right_mask] = [0.5, 0, 0]
        case "twist":
            left_points: pv.PointSet = pv.PointSet(surface.points[left_mask])
            left_points.rotate_x(-90, inplace=True)
            pin_disp[left_mask] = left_points.points - surface.points[left_mask]
            right_points: pv.PointSet = pv.PointSet(surface.points[right_mask])
            right_points.rotate_x(90, inplace=True)
            pin_disp[right_mask] = right_points.points - surface.points[right_mask]
    surface.point_data["pin_disp"] = pin_disp
    tetmesh = mkit.transfer.point_data.surface_to_tetmesh(surface, tetmesh)
    tetmesh.save(cfg.output)


if __name__ == "__main__":
    mkit.cli.run(main)
