import enum
import pathlib
from collections.abc import Mapping
from typing import Annotated, Optional

import pyvista as pv
import typer

from mesh_kit import cli as _cli
from mesh_kit import imagedata as _img
from mesh_kit.io import imagedata as _io
from mesh_kit.io import record as _record


class Component(enum.Enum):
    FACE = "face"
    SKULL = "skull"

    @property
    def threshold(self) -> float:
        return THRESHOLDS[self]


THRESHOLDS: Mapping[Component, float] = {
    Component.FACE: -128.0,
    Component.SKULL: 128.0,
}


def main(
    ct_path: Annotated[pathlib.Path, typer.Argument(exists=True)],
    output_file: Annotated[pathlib.Path, typer.Argument(dir_okay=False, writable=True)],
    *,
    component: Annotated[Component, typer.Option()],
    progress: Annotated[bool, typer.Option()] = True,
    record_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(exists=True, file_okay=False, writable=True),
    ] = None,
) -> None:
    data: pv.ImageData = _io.read(ct_path)
    data = _img.pad(data)
    _record.write(data, record_dir)
    data = data.gaussian_smooth(progress_bar=progress)
    _record.write(data, record_dir)
    contour: pv.PolyData = data.contour(
        isosurfaces=[component.threshold], progress_bar=progress
    )
    _record.write(contour, record_dir)
    contour = contour.connectivity(extraction_mode="largest", progress_bar=progress)
    contour.save(output_file)


if __name__ == "__main__":
    _cli.run(main)
