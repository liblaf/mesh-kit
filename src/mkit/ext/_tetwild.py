from typing import Any

import numpy as np
import numpy.typing as npt
import pyvista as pv

import mkit
import mkit.io
from mkit.ext._tetgen import tetgen
from mkit.logging import log_time
from mkit.typing import StrPath


@log_time
def tetwild(
    surface: Any,
    *,
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
    return tetmesh


@log_time
def tetwild_wrapped(
    outer: Any,
    inner: Any | None = None,
    *,
    executable: StrPath | None = None,
    lr: float | None = 0.05,
    stop_energy: float | None = None,
    coarsen: bool = False,
) -> pv.UnstructuredGrid:
    outer_pv: pv.PolyData = mkit.io.as_polydata(outer)
    surface: pv.PolyData
    if inner:
        inner_pv: pv.PolyData = mkit.io.as_polydata(inner)
        inner_pv.flip_normals()
        surface = pv.merge([outer_pv, inner_pv])
    else:
        surface = outer_pv
    mesh: pv.UnstructuredGrid = tetwild(
        surface,
        executable=executable,
        lr=lr,
        stop_energy=stop_energy,
        coarsen=coarsen,
    )
    surface_extracted: pv.PolyData = mesh.extract_surface(progress_bar=True)
    surface_extracted = surface_extracted.sample(
        surface, tolerance=1e-16, progress_bar=True, snap_to_closest_point=True
    )
    assert surface_extracted.point_data["vtkValidPointMask"].all()
    original_point_ids: npt.NDArray[np.int64] = surface_extracted.point_data[
        "vtkOriginalPointIds"
    ]
    for k, v in surface_extracted.point_data.items():
        mesh.point_data[k] = np.zeros((mesh.n_points, *v.shape[1:]), dtype=v.dtype)
        mesh.point_data[k][original_point_ids] = v
    mesh.field_data.update(surface_extracted.field_data)
    return mesh
