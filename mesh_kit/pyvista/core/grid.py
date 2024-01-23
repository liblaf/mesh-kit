import numpy as np
import pyvista as pv
from numpy import ma
from numpy import typing as npt


def pad(data: pv.ImageData) -> pv.ImageData:
    scalars: npt.NDArray = data.active_scalars.reshape(data.dimensions, order="F")
    padded_scalars: npt.NDArray = np.pad(
        array=scalars,
        pad_width=1,
        mode="constant",
        constant_values=ma.maximum_fill_value(scalars),
    )
    padded_data: pv.ImageData = pv.ImageData(
        dimensions=[dim + 2 for dim in data.dimensions],
        spacing=data.spacing,
        origin=data.origin,
    )
    padded_data.point_data.set_scalars(
        padded_scalars.flatten(order="F"), name=data.active_scalars_name
    )
    return padded_data
