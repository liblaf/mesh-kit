from typing import TypeVar

import pyvista as pv

from mkit.core.base._base import DataObjectBase

_T = TypeVar("_T", bound=pv.DataSet)


class ReprMixin(DataObjectBase[_T]): ...
