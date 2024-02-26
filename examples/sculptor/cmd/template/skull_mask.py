import pathlib
from typing import Annotated

import trimesh
import typer
from numpy import typing as npt
from scipy import spatial
from trimesh import smoothing

from mesh_kit import acvd as _acvd
from mesh_kit import cli as _cli
from mesh_kit import trimesh as _tri
from mesh_kit.io import trimesh as _io


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    output_file: Annotated[pathlib.Path, typer.Argument(dir_okay=False, writable=True)],
    *,
    acvd: Annotated[bool, typer.Option()] = True,
    smooth: Annotated[bool, typer.Option()] = False,
    threshold: Annotated[float, typer.Option(min=0.0)] = 0.02,
) -> None:
    mesh: trimesh.Trimesh
    attrs: dict[str, npt.NDArray]
    mesh, attrs = _io.read(input_file, attr=True)
    mandible: trimesh.Trimesh
    maxilla: trimesh.Trimesh
    mandible, maxilla = mesh.split()
    if acvd:
        mandible = _acvd.acvd(mandible)
    mandible = _tri.mesh_fix(mandible)
    if smooth:
        mandible = smoothing.filter_laplacian(mandible)
    tree: spatial.KDTree = maxilla.kdtree
    distance: npt.NDArray
    index: npt.NDArray
    distance, index = tree.query(
        mandible.vertices, distance_upper_bound=threshold * mesh.scale
    )
    vert_mask: npt.NDArray = distance < threshold * mesh.scale
    face_mask: npt.NDArray = _tri.mask.vert2face(mandible, vert_mask)
    mandible.update_faces(face_mask)
    _io.write(output_file, mandible, attr=True, **attrs)


if __name__ == "__main__":
    _cli.run(main)
