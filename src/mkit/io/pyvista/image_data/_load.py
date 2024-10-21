from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyvista as pv

if TYPE_CHECKING:
    import mkit.typing as mt


def load_image_data(path: mt.StrPath) -> pv.ImageData:
    path: Path = Path(path)
    if (path / "DIRFILE").exists():
        return pv.read(path, force_ext=".dcm")  # pyright: ignore [reportReturnType]
    return pv.read(path)  # pyright: ignore [reportReturnType]
