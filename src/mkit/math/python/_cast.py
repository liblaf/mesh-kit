from typing import Any

import mkit


def as_scalar(val: Any) -> float:
    if mkit.typing.jax.is_jax(val):
        return val.item()
    if mkit.typing.numpy.is_numpy(val):
        return val.item()
    if mkit.typing.torch.is_torch(val):
        return val.item()
    return val
