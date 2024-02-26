import pathlib
from typing import Annotated

import trimesh
import typer
from numpy import typing as npt
from trimesh import smoothing

from mesh_kit import acvd as _acvd
from mesh_kit import cli as _cli
from mesh_kit import trimesh as _tri
from mesh_kit.io import trimesh as _io


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    output_file: Annotated[pathlib.Path, typer.Argument(dir_okay=False, writable=True)],
    *,
    smooth: Annotated[bool, typer.Option()] = False,
    acvd: Annotated[bool, typer.Option()] = True,
) -> None:
    mesh: trimesh.Trimesh
    attrs: dict[str, npt.NDArray]
    mesh, attrs = _io.read(input_file, attr=True)
    mesh = mesh.split()[0]
    if acvd:
        mesh = _acvd.acvd(mesh)
    mesh = _tri.mesh_fix(mesh)
    if smooth:
        mesh = smoothing.filter_laplacian(mesh)
    _io.write(output_file, mesh, attr=True, **attrs)


if __name__ == "__main__":
    _cli.run(main)
