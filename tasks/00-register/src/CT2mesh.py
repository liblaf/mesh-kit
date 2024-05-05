import pathlib
from typing import Annotated

import mkit.cli
import mkit.io
import pyvista as pv
import typer


def main(
    ct_path: Annotated[pathlib.Path, typer.Argument(exists=True)],
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
    threshold: Annotated[float, typer.Option()],
) -> None:
    reader = pv.DICOMReader(ct_path)
    data: pv.ImageData = reader.read()
    data = data.gaussian_smooth(progress_bar=True)
    contour: pv.PolyData = data.contour([threshold], progress_bar=True)  # pyright: ignore [reportArgumentType]
    contour.connectivity("largest", inplace=True, progress_bar=True)
    contour.save(output_file)


if __name__ == "__main__":
    mkit.cli.run(main)
