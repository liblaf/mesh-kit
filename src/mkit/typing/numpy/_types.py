import numpy as np
from jaxtyping import Bool, Float, Integer

B = Bool[np.ndarray, ""]
BN = Bool[np.ndarray, "N"]

F = Float[np.ndarray, ""]
F3 = Float[np.ndarray, "3"]
F33 = Float[np.ndarray, "3 3"]
F34 = Float[np.ndarray, "3 4"]
F4 = Float[np.ndarray, "4"]
F43 = Float[np.ndarray, "4 3"]
F44 = Float[np.ndarray, "4 4"]
FMN = Float[np.ndarray, "M N"]
FMN3 = Float[np.ndarray, "M N 3"]
FN = Float[np.ndarray, "N"]
FN3 = Float[np.ndarray, "N 3"]
FNN = Float[np.ndarray, "N N"]

I = Integer[np.ndarray, ""]  # noqa: E741
I2 = Integer[np.ndarray, "2"]
I3 = Integer[np.ndarray, "3"]
I4 = Integer[np.ndarray, "4"]
IN = Integer[np.ndarray, "N"]
IN2 = Integer[np.ndarray, "N 2"]
IN3 = Integer[np.ndarray, "N 3"]
IN4 = Integer[np.ndarray, "N 4"]
