import numpy as np
from numpy import typing as npt


def vertex_to_face(
    faces: npt.ArrayLike, vert_mask: npt.ArrayLike
) -> npt.NDArray[np.bool_]:
    faces = np.asarray(faces)
    vert_mask = np.asarray(vert_mask)
    return faces[vert_mask].all(axis=1)
