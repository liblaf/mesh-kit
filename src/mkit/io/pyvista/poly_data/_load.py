from __future__ import annotations

from pathlib import Path

import pyvista as pv

import mkit.io as mi
import mkit.typing as mt


def load_poly_data(path: mt.StrPath) -> pv.PolyData:
    path: Path = Path(path)
    match path.suffix:
        case ".obj":
            return mi.pyvista.poly_data.load_obj(path)
        case _:
            return pv.read(path)  # pyright: ignore [reportReturnType]
