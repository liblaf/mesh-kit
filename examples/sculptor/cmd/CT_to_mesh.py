import enum
import pathlib
from typing import Annotated, Optional

import pyvista as pv
import typer

from mesh_kit.common import cli
from mesh_kit.io import ct, record
from mesh_kit.pyvista.core import grid


class Component(enum.StrEnum):
    FACE: str = "face"
    SKULL: str = "skull"


THRESHOLDS: dict[Component, int] = {
    Component.FACE: -128.0,
    Component.SKULL: 256.0,
}


def main(
    ct_path: Annotated[pathlib.Path, typer.Argument(exists=True)],
    output_path: Annotated[pathlib.Path, typer.Argument(dir_okay=False, writable=True)],
    component: Annotated[Component, typer.Option()],
    record_dir: Annotated[
        Optional[pathlib.Path], typer.Option("--record", exists=True, file_okay=False)
    ] = None,
) -> None:
    data: pv.ImageData = ct.read(ct_path)
    data: pv.ImageData = grid.pad(data)
    record.save(data, dir=record_dir)
    data: pv.ImageData = data.gaussian_smooth(progress_bar=True)
    record.save(data, dir=record_dir)
    contour: pv.PolyData = data.contour(
        isosurfaces=[THRESHOLDS[component]], progress_bar=True
    )
    record.save(contour, dir=record_dir)
    contour: pv.PolyData = contour.connectivity(
        extraction_mode="largest", progress_bar=True
    )
    contour.save(output_path)


if __name__ == "__main__":
    cli.run(main)
