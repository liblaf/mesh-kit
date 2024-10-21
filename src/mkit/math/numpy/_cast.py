from __future__ import annotations

import numpy as np
import numpy.typing as npt

import mkit.math as mm
import mkit.typing as mt


def as_bool(x: npt.ArrayLike) -> npt.NDArray[np.bool]:
    return as_type(x, bool)


def as_numpy(x: npt.ArrayLike) -> npt.NDArray[...]:
    if mt.is_torch(x):
        return x.numpy(force=True)
    return np.asarray(x)


def as_numpy_or_none(x: npt.ArrayLike | None) -> npt.NDArray[...] | None:
    if x is None:
        return None
    return as_numpy(x)


def as_type(x: npt.ArrayLike, dtype: npt.DTypeLike) -> npt.NDArray[...]:
    x: npt.NDArray[...] = as_numpy(x)
    dtype: np.dtype = np.dtype(dtype)
    if np.issubdtype(x.dtype, dtype):
        return x
    if np.isdtype(dtype, "bool"):
        if np.ptp(x) > 0:
            x = mm.numpy.scale(x)
        return x > 0.5
    if np.isdtype(dtype, "integral"):
        x = np.rint(x)
    return x.astype(dtype)
