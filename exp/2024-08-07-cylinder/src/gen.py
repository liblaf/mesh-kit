from typing import Literal

import mkit.cli
import mkit.ext
import mkit.transfer
import numpy as np
import numpy.typing as npt
import pyvista as pv


class Config(mkit.cli.BaseConfig):
    deform: Literal["squash", "stretch", "twist"]


@mkit.cli.cli(Config)
def main(cfg: Config) -> None:
    surface: pv.PolyData = pv.Cylinder()
    surface.triangulate(inplace=True, progress_bar=True)
    tetmesh: pv.UnstructuredGrid = mkit.ext.tetwild(surface)
    surface = tetmesh.extract_surface(progress_bar=True)
    left_mask: npt.NDArray[np.bool] = surface.points[:, 0] < surface.bounds[0] + 1e-3
    right_mask: npt.NDArray[np.bool] = surface.points[:, 0] > surface.bounds[1] - 1e-3
    surface.point_data["pin_mask"] = left_mask | right_mask
    pin_disp: npt.NDArray[np.floating] = np.zeros((surface.n_points, 3))
    match cfg.deform:
        case "squash":
            pin_disp[left_mask] = [0.25, 0, 0]
            pin_disp[right_mask] = [-0.25, 0, 0]
        case "stretch":
            pin_disp[left_mask] = [0.25, 0, 0]
            pin_disp[right_mask] = [-0.25, 0, 0]
        case "twist":
            raise NotImplementedError
    surface.point_data["pin_disp"] = pin_disp
    tetmesh = mkit.transfer.surface_to_tetmesh(surface, tetmesh)
    tetmesh.save("data/input.vtu")


if __name__ == "__main__":
    main()
