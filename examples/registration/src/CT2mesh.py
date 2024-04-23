import pathlib
from typing import Annotated

import pyvista as pv
import typer
from mkit import cli as _cli


def main(
    ct_path: Annotated[pathlib.Path, typer.Argument(exists=True)],
    threshold: Annotated[float, typer.Option()],
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
) -> None:
    reader = pv.DICOMReader(ct_path)
    ct: pv.ImageData = reader.read()
    ct = ct.gaussian_smooth(progress_bar=True)
    contour: pv.PolyData = ct.contour([threshold], progress_bar=True)  # pyright: ignore[reportArgumentType]
    contour.connectivity("largest", inplace=True, progress_bar=True)
    contour.save(output_file)


if __name__ == "__main__":
    _cli.run(main)
