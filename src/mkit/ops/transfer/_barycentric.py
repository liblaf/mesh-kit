from typing import Any

import numpy as np
import trimesh as tm
from jaxtyping import Shaped
from loguru import logger

import mkit
import mkit.typing.numpy as nt
from mkit.typing import AttributesLike


def point_to_point(
    source: Any,
    target: Any,
    data: AttributesLike | None = None,
    *,
    distance_threshold: float = 0.1,
) -> dict[str, Shaped[np.ndarray, "V ..."]]:
    if not data:
        return {}
    source: tm.Trimesh = mkit.io.trimesh.as_trimesh(source)
    target: tm.Trimesh = mkit.io.trimesh.as_trimesh(target)
    closest: nt.FN3
    dist: nt.FN
    triangle_id: nt.IN
    closest, dist, triangle_id = source.nearest.on_surface(target.vertices)
    faces: nt.IN3 = source.faces[triangle_id]
    barycentric: nt.FN3 = tm.triangles.points_to_barycentric(
        source.vertices[faces], closest
    )
    target_point_data: dict[str, Shaped[np.ndarray, "V ..."]] = {}
    valid: nt.BN = dist < distance_threshold * source.scale
    if not np.all(valid):
        logger.warning(
            "Some points are not within the distance threshold: {}",
            distance_threshold,
        )
    for k, v in data.items():
        data: Shaped[np.ndarray, "V 3 ..."] = np.asarray(v)[faces]  # (V, 3, ...)
        data_interp: Shaped[np.ndarray, "V 3 ..."] = np.einsum(
            "ij,ij...->i...", barycentric, data
        )
        target_point_data[k] = mkit.math.numpy.cast(data_interp, data.dtype)
        target_point_data[k][~valid] = np.nan
    return target_point_data


def cell_to_cell(
    source: Any,
    target: Any,
    data: AttributesLike | None = None,
    *,
    distance_threshold: float = 0.1,
) -> dict[str, Shaped[np.ndarray, "V ..."]]:
    if not data:
        return {}
    raise NotImplementedError
