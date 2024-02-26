import pathlib

import pyvista as pv

from mesh_kit.typing import check_type as _check_type


def read(path: pathlib.Path) -> pv.ImageData:
    if path.is_dir():
        reader = pv.DICOMReader(path)
        return _check_type(reader.read(), pv.ImageData)
    return _check_type(pv.read(path), pv.ImageData)
