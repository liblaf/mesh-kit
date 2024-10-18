from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import TYPE_CHECKING

import numpy.typing as npt

import mkit.math as m

if TYPE_CHECKING:
    import pyvista as pv

    import mkit.typing.numpy as tn


class Attrs(MutableMapping[str, npt.NDArray]):
    _data: pv.DataSetAttributes

    def __init__(self, data: pv.DataSetAttributes) -> None:
        self._data = data

    def __getitem__(self, key: str) -> npt.NDArray:
        return m.as_numpy(self._data[key])

    def __setitem__(self, key: str, value: tn.ArrayLike) -> None:
        self._data[key] = m.as_numpy(value)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._data.keys()

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        import prettytable

        table = prettytable.PrettyTable(
            [
                "Name",
                "Field",
                "Type",
                "N Comp",
                # "Min",
                # "Max",
            ]
        )
        table.set_style(prettytable.MSWORD_FRIENDLY)
        for k, v in self.items():
            table.add_row(
                [
                    k,
                    self._data.association.name,
                    v.dtype,
                    v.shape[1] if v.ndim > 1 else 1,
                ]
            )
        return table.get_string()
