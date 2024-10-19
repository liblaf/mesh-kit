from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import TYPE_CHECKING

import numpy.typing as npt

import mkit.math as mm

if TYPE_CHECKING:
    import pyvista as pv

    import mkit.typing.numpy as tn


class AttrsBase(MutableMapping[str, npt.NDArray]):
    _data: pv.DataSetAttributes

    def __init__(self, data: pv.DataSetAttributes) -> None:
        self._data = data

    def __getitem__(self, key: str) -> npt.NDArray:
        return mm.as_numpy(self._data[key])

    def __setitem__(self, key: str, value: tn.ArrayLike) -> None:
        self._data[key] = mm.as_numpy(value)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._data.keys()

    def __len__(self) -> int:
        return len(self._data)

    @property
    def association(self) -> pv.FieldAssociation:
        return self._data.association
