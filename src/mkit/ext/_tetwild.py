from typing import Any

import meshio

from mkit.typing import StrPath


def tetwild(
    surface: Any,
    *,
    executable: StrPath | None = None,
    lr: float | None = 0.05,
    stop_energy: float | None = None,
) -> meshio.Mesh:
    import os
    import pathlib
    import subprocess
    import tempfile

    import meshio

    import mkit.io

    surface_mesh: meshio.Mesh = mkit.io.as_meshio(surface)
    executable = os.getenv("TETWILD", executable) or "fTetWild"
    args: list[str] = []
    if lr is not None:
        args += ["--lr", str(lr)]
    if stop_energy is not None:
        args += ["--stop-energy", str(stop_energy)]
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = pathlib.Path(tmpdir_str)
        input_file: pathlib.Path = tmpdir / "mesh.ply"
        output_file: pathlib.Path = tmpdir / "mesh.msh"
        surface_mesh.write(input_file)
        subprocess.run(
            [executable, "--input", input_file, "--output", output_file, *args],
            stdin=subprocess.DEVNULL,
            check=True,
        )
        return meshio.read(output_file)
