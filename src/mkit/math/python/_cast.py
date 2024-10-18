from typing import Any

import mkit.typing as t


def as_scalar(val: Any) -> float:
    if t.is_jax(val):
        return val.item()
    if t.is_numpy(val):
        return val.item()
    if t.is_torch(val):
        return val.item()
    return val
