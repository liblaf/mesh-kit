from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv

    import mkit.typing.numpy as tn


class Attrs(MutableMapping[str, np.ndarray]):
    _data: pv.DataSetAttributes

    def __init__(self, data: pv.DataSetAttributes) -> None:
        self._data = data

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def __setitem__(self, key: str, value: tn.ArrayLike) -> None:
        self._data[key] = np.asarray(value)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._data.keys()

    def __len__(self) -> int:
        return len(self._data)
