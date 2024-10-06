import dataclasses
from typing import Any

import numpy as np
import numpy.typing as npt
from jaxtyping import Float, Integer

from mkit.ops.transfer._abc import C2CMethod, P2PMethod
from mkit.ops.transfer._barycentric import C2CBarycentric, P2PBarycentric
from mkit.ops.transfer._nearest import C2CNearest, P2PNearest
from mkit.typing import AttributeArray, AttributesLike

FloatAttributeArray = Float[np.ndarray, "N ..."]
IntAttributeArray = Integer[np.ndarray, "N ..."]


@dataclasses.dataclass(kw_only=True)
class C2CAuto(C2CMethod):
    distance_threshold: float = 0.1
    fill_value: npt.ArrayLike = np.nan

    def __call__(
        self,
        source: Any,
        target: Any,
        data: AttributesLike | None = None,
    ) -> dict[str, AttributeArray]:
        if not data:
            return {}
        float_data: dict[str, FloatAttributeArray]
        int_data: dict[str, IntAttributeArray]
        float_data, int_data = _split_data(data)
        return {
            **self.barycentric(source, target, float_data),
            **self.nearest(source, target, int_data),
        }

    @property
    def barycentric(self) -> C2CBarycentric:
        return C2CBarycentric(
            distance_threshold=self.distance_threshold, fill_value=self.fill_value
        )

    @property
    def nearest(self) -> C2CNearest:
        return C2CNearest(
            distance_threshold=self.distance_threshold, fill_value=self.fill_value
        )


@dataclasses.dataclass(kw_only=True)
class P2PAuto(P2PMethod):
    distance_threshold: float = 0.1
    fill_value: npt.ArrayLike = np.nan

    def __call__(
        self,
        source: Any,
        target: Any,
        data: AttributesLike | None = None,
    ) -> dict[str, AttributeArray]:
        if not data:
            return {}
        float_data: dict[str, FloatAttributeArray]
        int_data: dict[str, IntAttributeArray]
        float_data, int_data = _split_data(data)
        return {
            **self.barycentric(source, target, float_data),
            **self.nearest(source, target, int_data),
        }

    @property
    def barycentric(self) -> P2PBarycentric:
        return P2PBarycentric(
            distance_threshold=self.distance_threshold, fill_value=self.fill_value
        )

    @property
    def nearest(self) -> P2PNearest:
        return P2PNearest(
            distance_threshold=self.distance_threshold, fill_value=self.fill_value
        )


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
