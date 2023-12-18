import pathlib
from collections.abc import Sequence
from typing import Annotated, Optional

import numpy as np
import trimesh
import typer
from numpy import typing as npt
from scipy import interpolate

from mesh_kit.common import cli, path
from mesh_kit.registration import nricp


def main(
    source_filepath: Annotated[
        pathlib.Path, typer.Argument(exists=True, dir_okay=False)
    ],
    target_filepath: Annotated[
        pathlib.Path, typer.Argument(exists=True, dir_okay=False)
    ],
    *,
    output_filepath: Annotated[
        pathlib.Path, typer.Option("--output", dir_okay=False, writable=True)
    ],
    records_filepath: Annotated[
        Optional[pathlib.Path],
        typer.Option("--records", exists=True, file_okay=False, writable=True),
    ] = None,
    normal_weight: Annotated[float, typer.Option(min=0.0)] = 1.0,
    distance_threshold: Annotated[float, typer.Option(min=0.0)] = 0.1,
) -> None:
    source: trimesh.Trimesh = trimesh.load(source_filepath)
    target: trimesh.Trimesh = trimesh.load(target_filepath)
    source_positions: npt.NDArray = np.loadtxt(path.landmarks(source_filepath))
    source_landmarks: npt.NDArray
    _, source_landmarks = source.nearest.vertex(points=source_positions)
    target_positions: npt.NDArray = np.loadtxt(path.landmarks(target_filepath))
    # smoothness, landmark, normal, max_iter
    steps: Sequence[npt.ArrayLike] = interpolate.interp1d(
        x=range(4),
        y=[
            [0.01, 10, 2.0, 10],
            [0.02, 5, 2.0, 10],
            [0.03, 2.5, 2.0, 10],
            [0.01, 0, 0.0, 10],
        ],
        axis=0,
    )(np.linspace(start=0, stop=3, num=8))
    result: Sequence[npt.NDArray] = nricp.nricp_amberg(
        source_mesh=source,
        target_geometry=target,
        source_landmarks=source_landmarks,
        target_positions=target_positions,
        steps=steps,
        eps=1e-6,
        distance_threshold=distance_threshold,
        return_records=records_filepath is not None,
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
    source.export(output_filepath, encoding="ascii")


if __name__ == "__main__":
    cli.run(main)
