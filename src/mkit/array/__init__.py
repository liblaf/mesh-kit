import numpy as np
import numpy.typing as npt


def astype(_x: npt.ArrayLike, _dtype: npt.DTypeLike) -> npt.NDArray:
    x: npt.NDArray = np.asarray(_x)
    dtype: np.dtype = np.dtype(_dtype)
    if np.isdtype(dtype, ("bool", "integral")):
        x = x.round()
    return x.astype(dtype)
