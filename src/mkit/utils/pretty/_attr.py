from typing import Any

import numpy as np
import pydantic

import mkit.math as m
import mkit.typing as t


class SummaryAttr(pydantic.BaseModel):
    type: type | np.dtype
    value: Any | None = None


def summary_attr(value: Any) -> SummaryAttr:
    if t.is_array_like(value):
        value: np.ndarray = m.as_numpy(value)
        return SummaryAttr(type=value.dtype)
    return SummaryAttr(type=type(value))
