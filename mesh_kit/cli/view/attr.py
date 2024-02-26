import pathlib
from typing import Annotated

import pyvista as pv
import trimesh
import typer
from numpy import typing as npt
from pyvista import plotting

from mesh_kit import cli as _cli
from mesh_kit.io import trimesh as _io


def main(
    file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    key: Annotated[str, typer.Option()],
) -> None:
    mesh: trimesh.Trimesh
    attrs: dict[str, npt.NDArray]
    mesh, attrs = _io.read(file, attr=True)
    plotter = plotting.Plotter()
    values: npt.NDArray = attrs[key]
    match values.ndim:
        case 1:
            plotter.add_mesh(pv.wrap(mesh), scalars=values)
        case 2:
            plotter.add_mesh(pv.wrap(mesh))
            plotter.add_arrows(mesh.triangles_center, values)
        case _:
            raise ValueError
    plotter.show()


if __name__ == "__main__":
    _cli.run(main)
