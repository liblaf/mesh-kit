import pathlib

import numpy as np
from numpy import typing as npt


def read(mesh_path: pathlib.Path) -> npt.NDArray:
    return np.loadtxt(path(mesh_path))


def write(mesh_path: pathlib.Path, landmarks: npt.NDArray) -> None:
    np.savetxt(path(mesh_path), landmarks)


def path(mesh_path: pathlib.Path) -> pathlib.Path:
    return mesh_path.with_stem(mesh_path.stem + "-landmarks").with_suffix(".txt")
