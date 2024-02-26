import functools
import inspect
import pathlib
from collections import defaultdict

import numpy as np
import pydantic
import pyvista as pv
import trimesh

from mesh_kit.io import trimesh as _io_tri

_counters: defaultdict = defaultdict(int)


def _path(
    directory: pathlib.Path,
    idx: int,
    span: str | None = None,
    name: str | None = None,
    suffix: str | None = None,
) -> pathlib.Path:
    result: pathlib.Path = directory / f"{idx:02d}"
    if span:
        result = result.with_stem(f"{result.stem}-{span}")
    if name:
        result = result.with_stem(f"{result.stem}-{name}")
    if suffix:
        result = result.with_suffix(suffix)
    return result


def _write(
    data,
    directory: pathlib.Path,
    idx: int,
    span: str | None = None,
    name: str | None = None,
) -> None:
    path = functools.partial(_path, directory=directory, idx=idx, span=span, name=name)
    match data:
        case np.ndarray():
            np.save(path(suffix=".npy"), data)
        case pv.ImageData():
            data.save(path(suffix=".vtk"))
        case pv.PolyData():
            data.save(path(suffix=".ply"))
        case pydantic.BaseModel():
            path(suffix=".json").write_text(data.model_dump_json(indent=2))
        case trimesh.Trimesh():
            _io_tri.write(path(suffix=".ply"), data)
        case _:
            raise NotImplementedError


def write(
    data, directory: pathlib.Path | None, *, span: str | None = None, **kwargs
) -> None:
    if directory is None:
        return
    if span is None:
        frames: list[inspect.FrameInfo] = inspect.stack()
        frame: inspect.FrameInfo = frames[1]
        filename: pathlib.Path = pathlib.Path(frame.filename)
        span = f"{filename.stem}-{frame.function}-{frame.lineno}"
    write = functools.partial(
        _write, directory=directory, idx=_counters[directory], span=span
    )
    write(data)
    for k, v in kwargs.items():
        write(v, name=k)
    _counters[directory] += 1
