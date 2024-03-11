import pathlib
from typing import NamedTuple

import numpy as np
from numpy import typing as npt

from mesh_kit.io import utils as _utils


class Ele(NamedTuple):
    tetrahedra: npt.NDArray
    attr: npt.NDArray | None = None


def from_str(text: str, *, zero: bool = True) -> Ele:
    lines: list[str] = _utils.splitlines(text)
    # <# of tetrahedra> <nodes per tet. (4 or 10)> <region attribute (0 or 1)>
    num_tetrahedra: int
    nodes_per_tet: int
    region_attribute: int
    num_tetrahedra, nodes_per_tet, region_attribute = [int(s) for s in lines[0].split()]
    tetrahedra: npt.NDArray = np.full((num_tetrahedra, nodes_per_tet), -1, int)
    attr: npt.NDArray | None = (
        np.full((num_tetrahedra,), -1, int) if region_attribute else None
    )
    for line in lines[1:]:
        # <tetrahedron #> <node> <node> ... <node> [attribute]
        words: list[str] = line.split()
        idx: int = int(words[0])
        if not zero:
            idx -= 1
        tetrahedra[idx] = [int(s) for s in words[1:]]
        if region_attribute:
            assert attr is not None
            attr[idx] = int(words[-1])
    return Ele(tetrahedra, attr)


def load(file: pathlib.Path, *, zero: bool = True) -> Ele:
    return from_str(file.read_text(), zero=zero)
