import enum
import pathlib
from collections.abc import Sequence
from typing import Annotated, Optional

import numpy as np
import trimesh
import typer
from numpy import typing as npt

from mesh_kit.common import cli, path
from mesh_kit.registration import nricp


class Component(enum.Enum):
    FACE: str = "face"
    SKULL: str = "skull"


def main(
    source_filepath: Annotated[
        pathlib.Path, typer.Argument(exists=True, dir_okay=False)
    ],
    target_filepath: Annotated[
        pathlib.Path, typer.Argument(exists=True, dir_okay=False)
    ],
    *,
    component: Annotated[Component, typer.Option()],
    output_filepath: Annotated[
        pathlib.Path, typer.Option("--output", dir_okay=False, writable=True)
    ],
    record_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option("--records", exists=True, file_okay=False, writable=True),
    ] = None,
) -> None:
    source: trimesh.Trimesh = trimesh.load(source_filepath)
    target: trimesh.Trimesh = trimesh.load(target_filepath)
    source_positions: npt.NDArray = np.loadtxt(path.landmarks(source_filepath))
    source_landmarks: npt.NDArray
    _, source_landmarks = source.nearest.vertex(points=source_positions)
    target_positions: npt.NDArray = np.loadtxt(path.landmarks(target_filepath))
    match component:
        case Component.FACE:
            steps: Sequence[nricp.Params] = nricp.params_interp(
                x=np.linspace(start=0, stop=2, num=3),
                xp=[0, 1, 2],
                fp=[
                    nricp.Params(
                        weight_smooth=0.02,
                        weight_landmark=10,
                        weight_normal=0.5,
                        max_iter=10,
                        eps=1e-4,
                        distance_threshold=0.5,
                        correspondence_scale=1.1,
                        correspondence_weight_normal=0.5,
                    ),
                    nricp.Params(
                        weight_smooth=0.007,
                        weight_landmark=5,
                        weight_normal=0.5,
                        max_iter=10,
                        eps=1e-4,
                        distance_threshold=0.5,
                        correspondence_scale=1,
                        correspondence_weight_normal=0.5,
                    ),
                    nricp.Params(
                        weight_smooth=0.002,
                        weight_landmark=2.5,
                        weight_normal=0.5,
                        max_iter=10,
                        eps=1e-4,
                        distance_threshold=0.5,
                        correspondence_scale=1,
                        correspondence_weight_normal=0.5,
                    ),
                ],
            )
        case Component.SKULL:
            steps: Sequence[nricp.Params] = nricp.params_interp(
                x=np.linspace(start=0, stop=9, num=8),
                xp=[0, 4, 8, 9],
                fp=[
                    nricp.Params(
                        weight_smooth=0.01,
                        weight_landmark=10,
                        weight_normal=0.5,
                        max_iter=10,
                        eps=1e-6,
                        distance_threshold=0.5,
                        correspondence_scale=1,
                        correspondence_weight_normal=0.5,
                    ),
                    nricp.Params(
                        weight_smooth=0.02,
                        weight_landmark=5,
                        weight_normal=0.5,
                        max_iter=10,
                        eps=1e-6,
                        distance_threshold=0.5,
                        correspondence_scale=1,
                        correspondence_weight_normal=0.5,
                    ),
                    nricp.Params(
                        weight_smooth=0.03,
                        weight_landmark=2.5,
                        weight_normal=0.5,
                        max_iter=10,
                        eps=1e-6,
                        distance_threshold=0.5,
                        correspondence_scale=1,
                        correspondence_weight_normal=0.5,
                    ),
                    nricp.Params(
                        weight_smooth=0.01,
                        weight_landmark=0,
                        weight_normal=0,
                        max_iter=10,
                        eps=1e-6,
                        distance_threshold=0.5,
                        correspondence_scale=1,
                        correspondence_weight_normal=0.5,
                    ),
                ],
            )

        case _:
            assert False

    result: Sequence[npt.NDArray] = nricp.nricp_amberg(
        source_mesh=source,
        target_geometry=target,
        source_landmarks=source_landmarks,
        target_positions=target_positions,
        steps=steps,
        record_dir=record_dir,
    )

    source.vertices = result
    source.export(output_filepath, encoding="ascii")


if __name__ == "__main__":
    cli.run(main)
