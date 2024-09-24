import numpy as np
import numpy.typing as npt


def as_bool(x: npt.ArrayLike) -> npt.NDArray[np.bool]:
    return cast(x, bool)


def cast(x: npt.ArrayLike, dtype: npt.DTypeLike) -> npt.NDArray[...]:
    x: npt.NDArray[...] = np.asarray(x)
    dtype: np.dtype = np.dtype(dtype)
    if np.issubdtype(x.dtype, dtype):
        return x
    if np.isdtype(dtype, "bool"):
        x = np.interp(x, [np.min(x), np.max(x)], [0, 1])
        return x > 0.5
    if np.isdtype(dtype, ("integral")):
        x = np.rint(x)
    return x.astype(dtype)
