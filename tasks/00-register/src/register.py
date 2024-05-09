import pathlib
from typing import Annotated, Optional

import meshio
import mkit.cli
import mkit.io
import mkit.ops
import mkit.ops.register
import mkit.ops.register.nricp_amberg
import mkit.ops.register.nricp_amberg.nricp
import numpy as np
import trimesh
import typer
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
    mkit.cli.up_to_date(output_file, [__file__, source_file, target_file])
    source_io: meshio.Mesh = mkit.io.load_meshio(source_file)
    target_io: meshio.Mesh = mkit.io.load_meshio(target_file)
    source_landmarks: npt.NDArray[np.integer] = source_io.field_data.get("landmarks")
    target_landmarks: npt.NDArray[np.integer] = target_io.field_data.get("landmarks")
    source_attrs: dict[str, npt.NDArray] = (
        {"landmarks": source_landmarks} if source_landmarks is not None else {}
    )
    target_attrs: dict[str, npt.NDArray] = (
        {"landmarks": target_landmarks} if source_landmarks is not None else {}
    )
    if "register" in source_io.point_data:
        source_attrs["register"] = np.asarray(
            source_io.point_data["register"], np.float64
        )
    if "register" in target_io.point_data:
        target_attrs["register"] = np.asarray(
            target_io.point_data["register"], np.float64
        )
    source_tr: trimesh.Trimesh = mkit.io.as_trimesh(source_io)
    target_tr: trimesh.Trimesh = mkit.io.as_trimesh(target_io)
    result: npt.NDArray[np.floating] = (
        mkit.ops.register.nricp_amberg.nricp.nricp_amberg(
            source_tr,
            target_tr,
            source_attrs=source_attrs,
            target_attrs=target_attrs,
        )
    ).vertices
    source_io.points = result
    mkit.io.save(output_file, source_io)


if __name__ == "__main__":
    mkit.cli.run(main)
