import numpy as np
import numpy.typing as npt
import pyvista as pv

import mkit.ext
import mkit.transfer


def main() -> None:
    surface: pv.PolyData = pv.Box((-2, 2, -1, 1, -1, 1))
    surface.triangulate(inplace=True, progress_bar=True)
    tetmesh: pv.UnstructuredGrid = mkit.ext.tetwild(surface)
    surface = tetmesh.extract_surface(progress_bar=True)
    left_mask: npt.NDArray[np.bool] = surface.points[:, 0] < surface.bounds[0] + 1e-3
    right_mask: npt.NDArray[np.bool] = surface.points[:, 0] > surface.bounds[1] - 1e-3
    surface.point_data["pin_mask"] = left_mask | right_mask
    pin_disp: npt.NDArray[np.floating] = np.zeros((surface.n_points, 3))
    left_points: pv.PointSet = pv.PointSet(surface.points[left_mask])
    left_points_warp: pv.PointSet = left_points.rotate_x(-90)
    pin_disp[left_mask] = left_points_warp.points - left_points.points
    right_points: pv.PointSet = pv.PointSet(surface.points[right_mask])
    right_points_warp: pv.PointSet = right_points.rotate_x(90)
    pin_disp[right_mask] = right_points_warp.points - right_points.points
    surface.point_data["pin_disp"] = pin_disp
    tetmesh = mkit.transfer.surface_to_tetmesh(surface, tetmesh)
    tetmesh.save("data/input.vtu")


if __name__ == "__main__":
    main()
