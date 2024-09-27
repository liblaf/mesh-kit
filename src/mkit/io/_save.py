from pathlib import Path
from typing import Any

import mkit
from mkit.typing import StrPath


def save(mesh: Any, fpath: StrPath, *, ext: str | None = None) -> None:
    fpath: Path = Path(fpath)
    if ext is None:
        ext = fpath.suffix
    match ext:
        case ".obj":
            fpath.parent.mkdir(parents=True, exist_ok=True)
            mkit.io.pyvista.save_obj(mesh, fpath)
        case ".ply" | ".vtp":
            fpath.parent.mkdir(parents=True, exist_ok=True)
            mkit.io.pyvista.as_poly_data(mesh).save(fpath)
        case ".vtu":
            fpath.parent.mkdir(parents=True, exist_ok=True)
            mkit.io.pyvista.as_unstructured_grid(mesh).save(fpath)
        case _:
            msg: str = f"Unsupported file extension: {ext!r}"
            raise ValueError(msg)
