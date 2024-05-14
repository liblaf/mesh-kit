import pathlib
from typing import Annotated

import meshio
import mkit.cli
import mkit.io
import numpy as np
import scipy
import scipy.stats
import trimesh
import typer
from loguru import logger
from numpy import typing as npt


def main(
    predict_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    gt_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
) -> None:
    predict: meshio.Mesh = mkit.io.load_meshio(predict_file)
    gt: meshio.Mesh = mkit.io.load_meshio(gt_file)
    validation_mask: npt.NDArray[np.bool_] = np.asarray(
        predict.point_data["validation"], np.bool_
    )
    distance: npt.NDArray[np.floating] = loss(
        mkit.io.as_trimesh(predict),
        mkit.io.as_trimesh(gt),
        validation_mask=validation_mask,
    )
    logger.info("Mean Distance: {}", distance.mean())
    logger.info("SD: {}", distance.std())
    logger.info("50% Distance: {}", np.percentile(distance, 50))
    logger.info("90% Distance: {}", np.percentile(distance, 90))
    logger.info("95% Distance: {}", np.percentile(distance, 95))
    logger.info("99% Distance: {}", np.percentile(distance, 99))
    logger.info("2mm Score: {}", scipy.stats.percentileofscore(distance, 2.0))
    logger.info("Max Distance: {}", distance.max())
    distance_all: npt.NDArray[np.floating] = np.full(len(predict.points), np.nan)
    print(
        distance.mean(),
        distance.std(),
        np.percentile(distance, 50),
        np.percentile(distance, 90),
        np.percentile(distance, 95),
        np.percentile(distance, 99),
        scipy.stats.percentileofscore(distance, 2.0),
        distance.max(),
        sep=",",
    )
    distance_all[validation_mask] = distance
    mkit.io.save(output_file, predict, point_data={"loss": distance_all})


def loss(
    predict: trimesh.Trimesh,
    gt: trimesh.Trimesh,
    *,
    validation_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.floating]:
    logger.info("Extents: {}", predict.extents)
    closest: npt.NDArray[np.integer]
    distance: npt.NDArray[np.floating]
    triangle_id: npt.NDArray[np.integer]
    # distance, triangle_id = gt.nearest.vertex(predict.vertices[validation_mask])
    closest, distance, triangle_id = gt.nearest.on_surface(
        predict.vertices[validation_mask]
    )
    return distance


if __name__ == "__main__":
    mkit.cli.run(main)
