from typing import Any

import pyvista as pv

from mkit.logging import log_time
from mkit.typing import StrPath


@log_time
def tetwild(
    surface: Any,
    *,
    transfer: bool = False,
    executable: StrPath | None = None,
    lr: float | None = 0.05,
    stop_energy: float | None = None,
    coarsen: bool = False,
) -> pv.UnstructuredGrid:
    import os
    import pathlib
    import subprocess
    import tempfile

    import mkit.io
    import mkit.transfer.point_data

    surface_mesh: pv.PolyData = mkit.io.as_polydata(surface)
    executable = executable or os.getenv("TETWILD") or "fTetWild"
    args: list[str] = []
    if lr is not None:
        args += ["--lr", str(lr)]
    if stop_energy is not None:
        args += ["--stop-energy", str(stop_energy)]
    if coarsen:
        args += ["--coarsen"]
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = pathlib.Path(tmpdir_str)
        input_file: pathlib.Path = tmpdir / "mesh.ply"
        output_file: pathlib.Path = tmpdir / "mesh.msh"
        surface_mesh.save(input_file)
        subprocess.run(
            [executable, "--input", input_file, "--output", output_file, *args],
            stdin=subprocess.DEVNULL,
            check=True,
        )
        tetmesh: pv.UnstructuredGrid = pv.read(output_file)
    if transfer:
        tetmesh = mkit.transfer.point_data.surface_to_tetmesh(surface, tetmesh)
        tetmesh.field_data.update(surface.field_data)
    return tetmesh
