import pathlib
from typing import Annotated

import mkit.cli
import mkit.io
import mkit.ops
import mkit.ops.register
import mkit.ops.register.icp
import numpy as np
import trimesh
import typer
from loguru import logger
from numpy import typing as npt


def main(
    template_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    target_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    template: trimesh.Trimesh = mkit.io.load_trimesh(template_file)
    target: trimesh.Trimesh = mkit.io.load_trimesh(target_file)
    target_to_template: npt.NDArray[np.float64]
    cost: float
    target_to_template, cost = align(target, template)
    target.apply_transform(target_to_template)
    origin: npt.NDArray[np.float64] = np.mean(target.bounds, axis=0)
    origin[2] = target.bounds[1][2]
    normal: npt.NDArray[np.float64] = [0.0, 0.0, -1.0]
    end: npt.NDArray[np.float64] = origin + normal
    origin, end = trimesh.transform_points([origin, end], target_to_template)
    normal = end - origin
    normal /= np.linalg.norm(normal)
    logger.info("Plain Origin: {}", origin)
    logger.info("Plain Normal: {}", normal)


def align(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    initial: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.float64], float]:
    if initial is None:
        initial = trimesh.transformations.concatenate_matrices(
            trimesh.transformations.translation_matrix(target.centroid),
            trimesh.transformations.scale_matrix(target.scale),
            trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0]),
            trimesh.transformations.scale_matrix(1.0 / source.scale),
            trimesh.transformations.translation_matrix(-source.centroid),
        )
    source_to_target: npt.NDArray[np.float64]
    cost: float
    source_to_target, cost = mkit.ops.register.icp.icp(source, target, initial=initial)
    logger.info("ICP Cost: {}", cost)
    return source_to_target, cost


if __name__ == "__main__":
    mkit.cli.run(main)
