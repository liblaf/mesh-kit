from __future__ import annotations

import numpy as np

import mkit.math as mm
import mkit.typing.numpy as tn


def scale(x: tn.ArrayLike, a: float = 0, b: float = 1) -> np.ndarray:
    x: np.ndarray = mm.as_numpy(x)
    x = (x - x.min()) / np.ptp(x)
    x = x * (b - a) + a
    return x
