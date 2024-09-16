import numpy as np
import numpy.typing as npt


def as_bool(x: npt.ArrayLike) -> npt.NDArray[np.bool]:
    x: npt.NDArray = np.asarray(x)
    x = np.interp(x, [np.min(x), np.max(x)], [0, 1])
    return cast(x, bool)


def cast(x: npt.ArrayLike, dtype: npt.DTypeLike) -> npt.NDArray:
    x: npt.NDArray = np.asarray(x)
    dtype: np.dtype = np.dtype(dtype)
    if np.isdtype(dtype, ("bool", "integral")):
        x = np.rint(x)
    return x.astype(dtype)
