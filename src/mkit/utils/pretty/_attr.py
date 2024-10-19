from typing import Any

import numpy as np
import pydantic

import mkit.math as mm
import mkit.typing as mt


class SummaryAttr(pydantic.BaseModel):
    type: type | np.dtype
    value: Any | None = None


def summary_attr(value: Any) -> SummaryAttr:
    if mt.is_array_like(value):
        value: np.ndarray = mm.as_numpy(value)
        return SummaryAttr(type=value.dtype)
    return SummaryAttr(type=type(value))
