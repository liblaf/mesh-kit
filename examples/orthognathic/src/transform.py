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
    transform_file: Annotated[
        pathlib.Path, typer.Option("-t", "--transform", exists=True, dir_okay=False)
    ],
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False)
    ],
) -> None:
    io = pytorch3d.io.IO()
    source: structures.Meshes = io.load_mesh(source_file)
    transform: ops.points_alignment.SimilarityTransform = _transform.load(
        transform_file
    )
    source_verts: torch.Tensor = _a(source.verts_padded())
    result_verts: torch.Tensor = _transform.transform_points(transform, source_verts)
    result: structures.Meshes = structures.Meshes(
        verts=result_verts, faces=source.faces_padded()
    )
    io.save_mesh(result, output_file)


if __name__ == "__main__":
    _cli.run(main)
