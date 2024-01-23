import inspect
import pathlib
from collections import defaultdict
from collections.abc import MutableMapping, Sequence
from typing import Optional

import numpy as np
import pydantic
import pyvista as pv
import trimesh
from numpy import typing as npt

from mesh_kit.common import testing

counters: MutableMapping = defaultdict(int)


def save(
    data: trimesh.Trimesh | pv.PolyData | pv.ImageData,
    dir: Optional[pathlib.Path],
    *,
    landmarks: Optional[npt.NDArray] = None,
    params: Optional[pydantic.BaseModel] = None,
    source_positions: Optional[npt.NDArray] = None,
    target_positions: Optional[npt.NDArray] = None,
) -> None:
    if dir is None:
        return
    frames: Sequence[inspect.FrameInfo] = inspect.stack()
    frame: inspect.FrameInfo = frames[1]
    filename: pathlib.Path = pathlib.Path(frame.filename)
    id: str = f"{filename.stem}-{frame.function}-{frame.lineno}"
    match data:
        case trimesh.Trimesh():
            data.export(dir / f"{counters[dir]:02d}-{id}.ply")
        case pv.PolyData():
            data.save(dir / f"{counters[dir]:02d}-{id}.ply")
        case pv.ImageData():
            data.save(dir / f"{counters[dir]:02d}-{id}.vtk")
        case _:
            raise NotImplementedError()
    if landmarks is not None:
        testing.assert_shape(landmarks.shape, (-1, 3))
        np.savetxt(dir / f"{counters[dir]:02d}-{id}-landmarks.txt", landmarks)
    if params is not None:
        params_file: pathlib.Path = dir / f"{counters[dir]:02d}-{id}-params.json"
        params_file.write_text(params.model_dump_json())
    if source_positions is not None:
        testing.assert_shape(source_positions.shape, (-1, 3))
        np.savetxt(dir / f"{counters[dir]:02d}-{id}-source.txt", source_positions)
    if target_positions is not None:
        testing.assert_shape(target_positions.shape, (-1, 3))
        np.savetxt(dir / f"{counters[dir]:02d}-{id}-target.txt", target_positions)
    counters[dir] += 1
