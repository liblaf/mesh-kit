from __future__ import annotations

from typing import TYPE_CHECKING

import mkit.math as mm
import mkit.typing.numpy as tn

if TYPE_CHECKING:
    import numpy as np


def scale(x: tn.ArrayLike, a: float = 0, b: float = 1) -> np.ndarray:
    x: np.ndarray = mm.as_numpy(x)
    x = (x - x.min()) / (x.max() - x.min())
    x = x * (b - a) + a
    return x
