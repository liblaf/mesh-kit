from __future__ import annotations

import abc
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Self, TypeVar

import pyvista as pv

import mkit
import mkit.math as m
import mkit.typing as t
import mkit.typing.numpy as tn

if TYPE_CHECKING:
    import numpy as np

_T = TypeVar("_T", bound=pv.DataSet)


class DataObjectBase(Generic[_T], abc.ABC):
    name: str | None = None
    _data: _T
    _path: Path | None = None

    @property
    def point_data(self) -> mkit.Attrs:
        return mkit.Attrs(self._data.point_data)

    @point_data.setter
    def point_data(self, value: t.AttrsLike | None) -> None:
        self._data.point_data.clear()
        if value is not None:
            self._data.point_data.update({k: m.as_numpy(v) for k, v in value.items()})

    @property
    def cell_data(self) -> mkit.Attrs:
        return mkit.Attrs(self._data.cell_data)

    @cell_data.setter
    def cell_data(self, value: t.AttrsLike | None) -> None:
        self._data.cell_data.clear()
        if value is not None:
            self._data.cell_data.update({k: m.as_numpy(v) for k, v in value.items()})

    @property
    def field_data(self) -> mkit.Attrs:
        return mkit.Attrs(self._data.field_data)

    @field_data.setter
    def field_data(self, value: t.AttrsLike | None) -> None:
        self._data.field_data.clear()
        if value is not None:
            self._data.field_data.update({k: m.as_numpy(v) for k, v in value.items()})

    @property
    def area(self) -> float:
        return self._data.area

    @property
    def center(self) -> tn.FN3:
        return m.as_numpy(self._data.center)

    @property
    def n_points(self) -> int:
        return self._data.n_points

    @property
    def bounds(self) -> tn.Float[np.ndarray, "2 3"]:
        return m.as_numpy(self._data.bounds).reshape(3, 2).T

    @property
    def points(self) -> tn.FN3:
        return self._data.points

    @points.setter
    def points(self, value: tn.FN3Like) -> None:
        self._data.points = m.as_numpy(value)

    @property
    def path(self) -> Path | None:
        return self._path

    @classmethod
    def from_pyvista(cls, data: _T) -> Self:
        self: Self = cls()
        self._data = data
        return self

    @classmethod
    def load(cls, path: t.StrPath, ext: str | None = None) -> Self:
        path: Path = Path(path)
        if ext is None:
            ext = path.suffix
        if ext == ".obj":
            raise NotImplementedError
        data: _T = pv.read(path, force_ext=ext)  # pyright: ignore [reportAssignmentType]
        self: Self = cls.from_pyvista(data)
        self._path = Path(path)
        return self

    def save(self, path: t.StrPath, ext: str | None = None) -> None:
        path: Path = Path(path)
        if ext is None:
            ext = path.suffix
        if ext == ".obj":
            raise NotImplementedError
        path.parent.mkdir(parents=True, exist_ok=True)
        self._data.save(path)
