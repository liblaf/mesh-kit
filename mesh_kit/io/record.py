import functools
import inspect
import pathlib
from collections import defaultdict
from collections.abc import MutableMapping, Sequence
from typing import Optional

import numpy as np
import pydantic
import pyvista as pv
import trimesh

counters: MutableMapping = defaultdict(int)


def save(
    data: trimesh.Trimesh | pv.PolyData | pv.ImageData,
    directory: Optional[pathlib.Path],
    *,
    span: Optional[str] = None,
    params: Optional[pydantic.BaseModel] = None,
) -> None:
    if directory is None:
        return
    if span is None:
        frames: Sequence[inspect.FrameInfo] = inspect.stack()
        frame: inspect.FrameInfo = frames[1]
        filename: pathlib.Path = pathlib.Path(frame.filename)
        span = f"{filename.stem}-{frame.function}-{frame.lineno}"
    _path = functools.partial(
        path, directory=directory, index=counters[directory], span=span
    )
    match data:
        case np.ndarray():
            np.savetxt(_path(suffix=".txt"))
        case pv.ImageData():
            data.save(_path(suffix=".vtk"))
        case pv.PolyData():
            data.save(_path(suffix=".ply"))
        case trimesh.Trimesh():
            data.export(_path(suffix=".ply"))
        case _:
            raise NotImplementedError
    if params is not None:
        params_file: pathlib.Path = _path(name="params", suffix=".json")
        params_file.write_text(params.model_dump_json())
    counters[directory] += 1


def path(
    directory: pathlib.Path,
    index: int,
    span: Optional[str] = None,
    name: Optional[str] = None,
    suffix: Optional[str] = None,
) -> pathlib.Path:
    result: pathlib.Path = directory / f"{index:02d}"
    if span:
        result = result.with_stem(f"{result.stem}-{span}")
    if name:
        result = result.with_stem(f"{result.stem}-{name}")
    if suffix:
        result = result.with_suffix(suffix)
    return result
