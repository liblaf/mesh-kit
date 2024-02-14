import numpy as np
import pyvista as pv
from numpy import ma
from numpy import typing as npt


def pad(data: pv.ImageData) -> pv.ImageData:
    scalars: npt.NDArray = data.active_scalars.reshape(data.dimensions, order="F")
    scalars_pad: npt.NDArray = np.pad(
        scalars,
        pad_width=1,
        mode="constant",
        constant_values=ma.maximum_fill_value(scalars),
    )
    data_pad: pv.ImageData = pv.ImageData(
        dimensions=[dim + 2 for dim in data.dimensions],
        spacing=data.spacing,
        origin=data.origin,
    )
    data_pad.point_data.set_scalars(
        scalars=scalars_pad.flatten(order="F"), name=data.active_scalars_name
    )
    return data_pad
