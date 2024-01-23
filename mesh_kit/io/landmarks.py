import pathlib

import numpy as np
from numpy import typing as npt

from mesh_kit.common import path


def read(mesh_file: pathlib.Path) -> npt.NDArray:
    return np.loadtxt(path.landmarks(mesh_file))
