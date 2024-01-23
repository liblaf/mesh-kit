import inspect
import pathlib
from collections import defaultdict
from collections.abc import MutableMapping, Sequence

import pyvista as pv

counters: MutableMapping = defaultdict(int)


def save(data: pv.PolyData | pv.ImageData, dir: pathlib.Path) -> None:
    assert dir.is_dir()
    frames: Sequence[inspect.FrameInfo] = inspect.stack()
    frame: inspect.FrameInfo = frames[1]
    filename: pathlib.Path = pathlib.Path(frame.filename)
    id: str = f"{filename.stem}-{frame.function}-{frame.lineno}"
    match data:
        case pv.PolyData():
            data.save(dir / f"{counters[dir]:02d}-{id}.ply")
        case pv.ImageData():
            data.save(dir / f"{counters[dir]:02d}-{id}.vtk")
        case _:
            raise NotImplementedError()
    counters[dir] += 1
