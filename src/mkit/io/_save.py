from pathlib import Path
from typing import TYPE_CHECKING, Any

import mkit.io as mi
import mkit.typing as mt

if TYPE_CHECKING:
    import pyvista as pv


class UnsupportedFormatError(ValueError):
    ext: str

    def __init__(self, ext: str) -> None:
        super().__init__(f"Unsupported file format: {ext:r}")
        self.ext = ext


def save(
    path: mt.StrPath,
    data: Any,
    *,
    ext: str | None = None,
    point_data: mt.AttrsLike | None = None,
    cell_data: mt.AttrsLike | None = None,
    field_data: mt.AttrsLike | None = None,
) -> None:
    path: Path = Path(path)
    if ext is None:
        ext = path.suffix
    match ext:
        case ".obj" | ".ply" | ".vtp":
            data: pv.PolyData = mi.pyvista.as_poly_data(
                data, point_data=point_data, cell_data=cell_data, field_data=field_data
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            if ext == ".obj":
                mi.pyvista.poly_data.save_obj(path, data)
            else:
                data.save(path)
        case ".vtu":
            raise NotImplementedError
        case ".vti":
            raise NotImplementedError
        case _:
            raise UnsupportedFormatError(ext)
