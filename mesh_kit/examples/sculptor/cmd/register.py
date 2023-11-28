from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Optional, cast

import numpy as np
import trimesh
from numpy import typing as npt
from typer import Argument, Option

from mesh_kit.common import cli, path
from mesh_kit.registration import nricp


def main(
    source_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    target_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    *,
    output_filepath: Annotated[Path, Option("--output", dir_okay=False, writable=True)],
    records_filepath: Annotated[
        Optional[Path], Option("--records", exists=True, file_okay=False, writable=True)
    ] = None,
    normal_weight: Annotated[float, Option(min=0.0)] = 1.0,
    distance_threshold: Annotated[float, Option(min=0.0)] = 0.1,
) -> None:
    source: trimesh.Trimesh = trimesh.load(source_filepath)
    target: trimesh.Trimesh = trimesh.load(target_filepath)
    source_positions: npt.NDArray = np.loadtxt(path.landmarks_filepath(source_filepath))
    source_landmarks: npt.NDArray
    _, source_landmarks = source.nearest.vertex(points=source_positions)
    target_positions: npt.NDArray = np.loadtxt(path.landmarks_filepath(target_filepath))
    # smoothness, landmark, normal, max_iter
    steps: Sequence[npt.ArrayLike] = [
        [0.03, 10, 0.4, 100],
        [0.02, 5, 0.6, 100],
        [0.01, 2.5, 0.8, 100],
        [0.001, 0, 1.0, 100],
    ]
    result: Sequence[npt.NDArray] = cast(
        npt.NDArray,
        nricp.nricp_amberg(
            source_mesh=source,
            target_geometry=target,
            source_landmarks=source_landmarks,
            target_positions=target_positions,
            steps=steps,
            eps=1e-5,
            distance_threshold=distance_threshold,
            return_records=records_filepath is not None,
        ),
    )
    if records_filepath is None:
        source.vertices = result
    else:
        for i, record in enumerate(result):
            source.vertices = record
            source.export(records_filepath / f"{i:03d}.ply")
            np.savetxt(
                records_filepath / f"{i:03d}-source-landmarks.txt",
                record[source_landmarks],
            )
            np.savetxt(
                records_filepath / f"{i:03d}-target-landmarks.txt",
                target_positions,
            )
        source.vertices = result[-1]
    source.export(output_filepath)


if __name__ == "__main__":
    cli.run(main)
