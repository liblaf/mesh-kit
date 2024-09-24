from typing import Any

import numpy as np
import numpy.typing as npt
from jaxtyping import Float, Integer

import mkit.ops.transfer._barycentric as barycentric
import mkit.ops.transfer._nearest as nearest
from mkit.typing import AttributeArray, AttributesLike

FloatAttributeArray = Float[np.ndarray, "N ..."]
IntAttributeArray = Integer[np.ndarray, "N ..."]


def point_to_point(
    source: Any,
    target: Any,
    data: AttributesLike | None = None,
    *,
    distance_threshold: float = 0.1,
) -> dict[str, AttributeArray]:
    if not data:
        return {}
    float_data: dict[str, FloatAttributeArray]
    int_data: dict[str, IntAttributeArray]
    float_data, int_data = _split_data(data)
    return {
        **barycentric.point_to_point(
            source, target, float_data, distance_threshold=distance_threshold
        ),
        **nearest.point_to_point(
            source, target, int_data, distance_threshold=distance_threshold
        ),
    }


def cell_to_cell(
    source: Any,
    target: Any,
    data: AttributesLike | None = None,
    *,
    distance_threshold: float = 0.1,
) -> dict[str, AttributeArray]:
    if not data:
        return {}
    float_data: dict[str, FloatAttributeArray]
    int_data: dict[str, IntAttributeArray]
    float_data, int_data = _split_data(data)
    return {
        **barycentric.cell_to_cell(
            source, target, float_data, distance_threshold=distance_threshold
        ),
        **nearest.cell_to_cell(
            source, target, int_data, distance_threshold=distance_threshold
        ),
    }


def _split_data(
    data: AttributesLike | None = None,
) -> tuple[dict[str, FloatAttributeArray], dict[str, IntAttributeArray]]:
    if not data:
        return {}, {}
    float_data: dict[str, FloatAttributeArray] = {}
    int_data: dict[str, IntAttributeArray] = {}
    for k, v in data.items():
        value: npt.NDArray[...] = np.asarray(v)
        if np.isdtype(value.dtype, "integral"):
            int_data[k] = value
        else:
            float_data[k] = value
    return float_data, int_data
