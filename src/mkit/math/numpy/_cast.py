import numpy as np
import numpy.typing as npt


def cast(x: npt.ArrayLike, dtype: npt.DTypeLike) -> npt.NDArray:
    x: npt.NDArray = np.asarray(x)
    dtype: np.dtype = np.dtype(dtype)
    if np.isdtype(dtype, ("bool", "integral")):
        x = np.rint(x)
    return x.astype(dtype)
