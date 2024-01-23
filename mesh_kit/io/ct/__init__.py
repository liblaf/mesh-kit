import pathlib

import pyvista as pv


def read(path: pathlib.Path) -> pv.ImageData:
    if path.is_dir():
        reader = pv.DICOMReader(path)
        return reader.read()
    return pv.read(path)
