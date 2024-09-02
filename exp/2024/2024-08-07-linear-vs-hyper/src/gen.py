import pathlib

import mkit.cli
import mkit.ext
import mkit.transfer.point_data
import numpy as np
import numpy.typing as npt
import pyvista as pv


class Config(mkit.cli.BaseConfig):
    stretch: float
    output: pathlib.Path


def main(cfg: Config) -> None:
    surface: pv.PolyData = pv.Cylinder()
    surface.triangulate(inplace=True, progress_bar=True)
    tetmesh: pv.UnstructuredGrid = mkit.ext.tetwild(surface)
    surface = tetmesh.extract_surface(progress_bar=True)
    left_mask: npt.NDArray[np.bool] = surface.points[:, 0] < surface.bounds[0] + 1e-3
    right_mask: npt.NDArray[np.bool] = surface.points[:, 0] > surface.bounds[1] - 1e-3
    surface.point_data["pin_mask"] = left_mask | right_mask
    pin_disp: npt.NDArray[np.floating] = np.zeros((surface.n_points, 3))
    pin_disp[left_mask] = [-cfg.stretch, 0, 0]
    pin_disp[right_mask] = [cfg.stretch, 0, 0]
    surface.point_data["pin_disp"] = pin_disp
    tetmesh = mkit.transfer.point_data.surface_to_tetmesh(surface, tetmesh)
    tetmesh.save(cfg.output)


if __name__ == "__main__":
    mkit.cli.run(main)
