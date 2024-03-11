import functools
import io
import pathlib
from typing import NamedTuple

import numpy as np
from numpy import typing as npt

from mesh_kit.io import utils as _utils


class Node(NamedTuple):
    points: npt.NDArray
    attrs: npt.NDArray | None = None
    boundary_marker: npt.NDArray | None = None


def from_str(text: str, *, zero: bool = True) -> Node:
    lines: list[str] = _utils.splitlines(text)
    # <# of points> <dimension (3)> <# of attributes> <boundary markers (0 or 1)>
    num_points: int
    dimension: int
    num_attrs: int
    boundary_markers: int
    num_points, dimension, num_attrs, boundary_markers = [
        int(s) for s in lines[0].split()
    ]
    assert dimension == 3
    points: npt.NDArray = np.full((num_points, dimension), np.nan)
    attrs: npt.NDArray | None = (
        np.full((num_points, num_attrs), np.nan) if num_attrs > 0 else None
    )
    boundary_marker: npt.NDArray | None = (
        np.zeros((num_points,), dtype=int) if boundary_markers else None
    )
    for line in lines[1:]:
        # <point #> <x> <y> <z> [attributes] [boundary marker]
        words: list[str] = line.split()
        i = 0
        point_idx = int(words[i])
        if not zero:
            point_idx -= 1
        i = 1
        points[point_idx] = [float(s) for s in words[i : i + dimension]]
        i += dimension
        if num_attrs > 0:
            assert attrs is not None
            attrs[point_idx] = [float(s) for s in words[i : i + num_attrs]]
            i += num_attrs
        if boundary_markers:
            assert boundary_marker is not None
            boundary_marker[i] = int(words[i])
    return Node(points, attrs, boundary_marker)


def load(file: pathlib.Path, *, zero: bool = True) -> Node:
    return from_str(file.read_text(), zero=zero)


def to_str(
    points: npt.NDArray,
    *,
    attrs: npt.NDArray | None = None,
    boundary_marker: npt.NDArray | None = None,
    zero: bool = True,
) -> str:
    num_points: int
    dimension: int
    num_points, dimension = points.shape
    assert dimension == 3
    num_attrs: int = attrs.shape[1] if attrs is not None else 0
    boundary_markers = 1 if boundary_marker is not None else 0
    fp = io.StringIO()
    fprint = functools.partial(print, file=fp)
    fprint(
        "# <# of points> <dimension (3)> <# of attributes> <boundary markers (0 or 1)>"
    )
    fprint(num_points, dimension, num_attrs, boundary_markers)
    fprint("# <point #> <x> <y> <z> [attributes] [boundary marker]")
    for i, point in enumerate(points):
        if not zero:
            i += 1
        fprint(i, *point, end="")
        if attrs is not None:
            fprint("", *attrs[i], end="")
        if boundary_marker is not None:
            fprint("", boundary_marker[i], end="")
        fprint()
    return fp.getvalue()


def save(
    file: pathlib.Path,
    points: npt.NDArray,
    *,
    attrs: npt.NDArray | None = None,
    boundary_marker: npt.NDArray | None = None,
    zero: bool = True,
) -> None:
    file.write_text(
        to_str(points, attrs=attrs, boundary_marker=boundary_marker, zero=zero)
    )
