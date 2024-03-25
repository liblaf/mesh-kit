import pathlib
from typing import Annotated

import pytorch3d.io
import torch
import typer
from mkit import cli as _cli
from mkit.pytorch3d import transform as _transform
from mkit.typing import as_any as _a
from pytorch3d import ops, structures


def main(
    source_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    target_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False)
    ],
) -> None:
    io = pytorch3d.io.IO()
    source_mesh: structures.Meshes = io.load_mesh(source_file)
    target_mesh: structures.Meshes = io.load_mesh(target_file)
    source_verts: torch.Tensor = _a(source_mesh.verts_padded())
    target_verts: torch.Tensor = _a(target_mesh.verts_padded())
    transform: ops.points_alignment.SimilarityTransform = (
        ops.corresponding_points_alignment(
            source_verts, target_verts, estimate_scale=False, allow_reflection=False
        )
    )
    _transform.save(output_file, transform)


if __name__ == "__main__":
    _cli.run(main)
