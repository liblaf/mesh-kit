from typing import TypeVar

import pyvista as pv

from mkit.core.base._repr import ReprMixin

_T = TypeVar("_T", bound=pv.DataSet)


class DataObject(ReprMixin[_T]): ...
