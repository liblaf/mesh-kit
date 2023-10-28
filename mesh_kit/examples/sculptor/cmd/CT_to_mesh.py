import functools
import os
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import cv2 as cv
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import trimesh.util
import trimesh.voxel.ops
from nrrd import NRRDHeader
from numpy.typing import NDArray
from trimesh import Trimesh
from trimesh.points import PointCloud
from typer import Argument, Option

from mesh_kit.common.cli import run

BOUNDARY: float = 0.30


class Method(str, Enum):
    EDGE_DETECTION = "edge-detection"
    MARCHING_CUBES = "marching-cubes"


def save_artifact(artifact: Optional[Path], step: str, z: int, data: NDArray) -> None:
    if artifact is None:
        return
    data = to_uint8(data=data)
    os.makedirs(name=artifact / step, exist_ok=True)
    plt.imsave(fname=str(artifact / step / f"{z:03d}.png"), arr=data)


def to_uint8(data: NDArray) -> NDArray:
    data = np.interp(data, (data.min(), data.max()), (0, 255))
    return data.astype(np.uint8)


def process(
    z: int,
    data: NDArray,
    threshold: float = 0.0,
    *,
    artifact: Optional[Path] = None,
    method: Method = Method.MARCHING_CUBES,
    morph_kernel_size: int = 1,
) -> NDArray:
    save_artifact(artifact=artifact, step="00-raw", z=z, data=data)
    threshold, data = cv.threshold(
        src=data, thresh=threshold, maxval=1.0, type=cv.THRESH_BINARY_INV
    )
    save_artifact(artifact=artifact, step="01-threshold", z=z, data=data)
    data = cv.morphologyEx(
        src=data,
        op=cv.MORPH_OPEN,
        kernel=np.ones(shape=(morph_kernel_size, morph_kernel_size)),
    )
    save_artifact(artifact=artifact, step="02-open", z=z, data=data)
    num_components: int
    labels: NDArray
    stats: NDArray
    centroids: NDArray
    num_components, labels, stats, centroids = cv.connectedComponentsWithStats(
        image=to_uint8(data)
    )
    background_label: int = np.argmax(
        stats[:, cv.CC_STAT_WIDTH] * stats[:, cv.CC_STAT_HEIGHT]  # type: ignore
    )
    data[labels == background_label] = 0
    data[labels != background_label] = 1
    save_artifact(artifact=artifact, step="03-background", z=z, data=data)
    num_components, labels, stats, centroids = cv.connectedComponentsWithStats(
        image=to_uint8(data)
    )
    for i in range(num_components):
        if not (centroids[i, 0] < (1.0 - BOUNDARY) * data.shape[0]):
            data[labels == i] = 0
    save_artifact(artifact=artifact, step="04-grid", z=z, data=data)
    if method == Method.MARCHING_CUBES:
        return data.astype(bool)
    data = cv.Canny(image=data, threshold1=64.0, threshold2=128.0)
    save_artifact(artifact=artifact, step="05-edge", z=z, data=data)
    if method == Method.EDGE_DETECTION:
        return data.astype(bool)


def main(
    input_path: Annotated[Path, Argument(exists=True, dir_okay=False)],
    output_path: Annotated[Path, Argument(dir_okay=False, writable=True)],
    *,
    artifact: Annotated[Optional[Path], Option()] = None,
    method: Annotated[Method, Option()] = Method.MARCHING_CUBES,
    morph_kernel_size: Annotated[int, Option()] = 1,
    threshold: Annotated[float, Option(help="Face: 0.0, Skull: 250.0")] = 0.0,
) -> None:
    data: NDArray
    header: NRRDHeader
    data, header = nrrd.read(filename=str(input_path))
    with ProcessPoolExecutor() as executor:
        data: NDArray = np.array(
            list(
                executor.map(
                    functools.partial(
                        process,
                        threshold=threshold,
                        artifact=artifact,
                        method=method,
                        morph_kernel_size=morph_kernel_size,
                    ),
                    range(data.shape[2]),
                    np.moveaxis(data, source=2, destination=0),
                )
            )
        )
        data = np.moveaxis(data, source=0, destination=2)
    points: NDArray = np.array(np.where(data))
    points = np.moveaxis(points, source=0, destination=1)
    mesh: PointCloud | Trimesh
    match method:
        case Method.EDGE_DETECTION:
            mesh = PointCloud(vertices=points)
        case Method.MARCHING_CUBES:
            mesh = trimesh.voxel.ops.points_to_marching_cubes(points=points)
    transform_matrix: NDArray = np.pad(
        header["space directions"], pad_width=((0, 1), (0, 1))
    )
    transform_matrix[3, 3] = 1.0
    mesh.apply_transform(transform_matrix)
    mesh.apply_translation(header["space origin"])
    mesh.export(output_path)


if __name__ == "__main__":
    run(main)
