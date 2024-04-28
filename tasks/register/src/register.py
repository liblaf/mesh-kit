import functools
import pathlib
from typing import Annotated, Optional

import meshio
import numpy as np
import trimesh
import typer
from mkit import cli
from mkit import io as _io
from numpy import typing as npt


def main(
    source_file: Annotated[pathlib.Path, typer.Argument(exists=True)],
    target_file: Annotated[pathlib.Path, typer.Argument(exists=True)],
    *,
    output_file: Annotated[pathlib.Path, typer.Option("-o", "--output", writable=True)],
    record_dir: Annotated[
        Optional[pathlib.Path], typer.Option(file_okay=False, writable=True)
    ] = None,
) -> None:
    source_io: meshio.Mesh = meshio.read(source_file)
    target_io: meshio.Mesh = meshio.read(target_file)
    source_tr: trimesh.Trimesh = _io.to_trimesh(source_io)
    target_tr: trimesh.Trimesh = _io.to_trimesh(target_io)
    source_effective_point_mask = source_io.point_data["effective"]
    raise NotImplementedError


def register(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    source_effective_vert_mask: npt.NDArray[np.bool_] | None = None,
    target_effective_vert_mask: npt.NDArray[np.bool_] | None = None,
) -> None:
    scale: float = source.scale
    centroid: npt.NDArray[np.float64] = source.centroid
    _normalize = functools.partial(normalize, centroid=centroid, scale=scale)
    _denormalize = functools.partial(denormalize, centroid=centroid, scale=scale)
    source = _normalize(source)
    target = _normalize(target)
    source_mask: npt.NDArray[np.bool_] = effective_mask(source, target)
    target_mask: npt.NDArray[np.bool_] = effective_mask(target, source)
    raise NotImplementedError


def effective_mask(
    mesh: trimesh.Trimesh, other: trimesh.Trimesh, *, threshold: float = 0.1
) -> npt.NDArray[np.bool_]:
    closest: npt.NDArray[np.float64]
    distance: npt.NDArray[np.float64]
    triangle_id: npt.NDArray[np.int64]
    closest, distance, triangle_id = other.nearest.on_surface(mesh.vertices)
    mask: npt.NDArray[np.bool_] = distance < threshold
    return mask


def normalize(
    mesh: trimesh.Trimesh, *, centroid: npt.NDArray[np.float64], scale: float
) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_scale(1 / scale)
    mesh.apply_translation(-centroid)
    return mesh


def denormalize(
    mesh: trimesh.Trimesh, *, centroid: npt.NDArray[np.float64], scale: float
) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_translation(centroid)
    mesh.apply_scale(scale)
    return mesh


if __name__ == "__main__":
    cli.run(main)
