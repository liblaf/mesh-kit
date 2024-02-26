import pathlib
from collections.abc import Mapping
from typing import Literal, overload

import numpy as np
import trimesh
from numpy import typing as npt

from mesh_kit.typing import typechecked as _typechecked


@overload
def read(file: pathlib.Path, *, attr: Literal[False] = False) -> trimesh.Trimesh:
    ...


@overload
def read(
    file: pathlib.Path, *, attr: Literal[True]
) -> tuple[trimesh.Trimesh, dict[str, npt.NDArray]]:
    ...


@_typechecked
def read(
    file: pathlib.Path, *, attr: bool = False
) -> tuple[trimesh.Trimesh, dict[str, npt.NDArray]] | trimesh.Trimesh:
    mesh: trimesh.Trimesh = trimesh.load(file)
    if attr:
        if (data_file := file.with_suffix(".npz")).exists():
            data: Mapping[str, npt.NDArray] = np.load(data_file)
            attrs: dict[str, npt.NDArray] = {}
            for k, v in data.items():
                if k.startswith("vert:"):
                    mesh.vertex_attributes[k.removeprefix("vert:")] = v
                elif k.startswith("face:"):
                    mesh.face_attributes[k.removeprefix("face:")] = v
                attrs[k] = v
            return mesh, attrs
        return mesh, {}
    return mesh


@_typechecked
def write(
    file: pathlib.Path,
    mesh: trimesh.Trimesh,
    *,
    attr: bool = True,
    **kwargs: npt.ArrayLike,
) -> None:
    mesh.export(file)
    if attr:
        data: dict[str, npt.ArrayLike] = {}
        for k, v in mesh.vertex_attributes.items():
            data[f"vert:{k}"] = v
        for k, v in mesh.face_attributes.items():
            data[f"face:{k}"] = v
        data.update(kwargs)
        if data:
            np.savez_compressed(file.with_suffix(".npz"), **data)
