from typing import Any

import mkit.typing as mt


def as_scalar(val: Any) -> float:
    if mt.is_jax(val):
        return val.item()
    if mt.is_numpy(val):
        return val.item()
    if mt.is_torch(val):
        return val.item()
    return val
