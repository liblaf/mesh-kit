import numpy as np
import trimesh
from numpy import typing as npt


def find_inner_point(
    mesh: trimesh.Trimesh, *, max_retry: int = 8
) -> npt.NDArray[np.float64]:
    for _ in range(max_retry):
        origin: npt.NDArray[np.float64] = mesh.bounds[0]
        end_point: npt.NDArray[np.float64] = mesh.sample(1, return_index=False)[0]
        direction: npt.NDArray[np.float64] = end_point - origin
        locations: npt.NDArray[np.float64]
        index_ray: npt.NDArray[np.intp]
        index_tri: npt.NDArray[np.intp]
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            [origin], [direction]
        )
        if len(locations) < 2:
            continue
        result: npt.NDArray = (locations[0] + locations[1]) / 2
        if mesh.contains([result])[0]:
            return result
    raise ValueError("Cannot find inner point")
