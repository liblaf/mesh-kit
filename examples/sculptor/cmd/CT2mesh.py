import enum
import pathlib
from collections.abc import Mapping
from typing import Annotated, Optional

import pyvista as pv
import typer

from mesh_kit.common import cli as _cli
from mesh_kit.io import ct as _ct
from mesh_kit.io import record as _record
from mesh_kit.pyvista.core import grid as _grid


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
    record_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(exists=True, file_okay=False, writable=True),
    ] = None,
) -> None:
    data: pv.ImageData = _ct.read(ct_path)
    data = _grid.pad(data)
    _record.save(data, record_dir)
    data = data.gaussian_smooth(progress_bar=True)
    _record.save(data, record_dir)
    contour: pv.PolyData = data.contour(
        isosurfaces=[component.threshold], progress_bar=True
    )
    _record.save(contour, record_dir)
    contour: pv.PolyData = contour.connectivity(
        extraction_mode="largest", progress_bar=True
    )
    contour.save(output_file)


if __name__ == "__main__":
    _cli.run(main)
