import pathlib
from typing import Annotated

import mkit.cli
import mkit.io
import numpy as np
import pyvista as pv
import typer
from numpy import typing as npt


def main(
    ct_path: Annotated[pathlib.Path, typer.Argument(exists=True)],
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
    threshold: Annotated[float, typer.Option()],
) -> None:
    mkit.cli.up_to_date(output_file, [__file__, ct_path])
    reader = pv.DICOMReader(ct_path)
    data: pv.ImageData = reader.read()
    data = data.gaussian_smooth(progress_bar=True)
    data = pad(data)
    contour: pv.PolyData = data.contour([threshold], progress_bar=True)  # pyright: ignore [reportArgumentType]
    contour.connectivity("largest", inplace=True, progress_bar=True)
    contour.save(output_file)


def pad(data: pv.ImageData) -> pv.ImageData:
    name: str = data.active_scalars_name
    scalars: npt.NDArray = data.active_scalars
    scalars = scalars.reshape(data.dimensions, order="F")
    scalars_padded: npt.NDArray = np.pad(
        scalars, 1, mode="constant", constant_values=np.iinfo(scalars.dtype).min
    )
    data_padded = pv.ImageData(
        dimensions=scalars_padded.shape,
        spacing=data.spacing,
        origin=data.origin,
    )
    data_padded.point_data[name] = scalars_padded.flatten(order="F")  # pyright: ignore [reportAttributeAccessIssue]
    data_padded.set_active_scalars(name, preference="point")
    return data_padded


if __name__ == "__main__":
    mkit.cli.run(main)
