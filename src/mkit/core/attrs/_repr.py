from __future__ import annotations

import dataclasses
from typing import Generic, TypeVar

import numpy as np
import prettytable
import pyvista as pv

import mkit.math as mm
import mkit.typing as mt
import mkit.typing.numpy as tn
from mkit.core.attrs._base import AttrsBase

_T = TypeVar("_T", bound=mt.Scalar)


@dataclasses.dataclass(kw_only=True)
class DescribeResult(Generic[_T]):
    """...

    References:
        1. <https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.describe.html>
        2. <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html>
    """

    count: int
    unique: int
    top: _T | None = None
    freq: int
    mean: _T
    min: _T
    p25: _T | None = None
    p50: _T | None = None
    p75: _T | None = None
    max: _T


def describe_array(arr: tn.ArrayLike) -> DescribeResult:
    arr: np.ndarray = mm.as_numpy(arr)
    values: tn.NDArray
    counts: tn.NDArray[np.intp]
    values, counts = np.unique(arr, return_counts=True, axis=0)
    if arr.ndim == 2:
        arr = np.linalg.norm(arr, axis=1)
        values = np.linalg.norm(values, axis=1)
    result = DescribeResult(
        count=np.count_nonzero(~np.isnan(arr)),
        unique=len(values),
        top=values[counts.argmax()],
        freq=counts.max(),
        mean=np.nanmean(arr),
        min=np.nanmin(arr),
        max=np.nanmax(arr),
    )
    if not np.isdtype(arr.dtype, "bool"):
        result.p25 = np.nanpercentile(arr, 25)
        result.p50 = np.nanpercentile(arr, 50)
        result.p75 = np.nanpercentile(arr, 75)
    return result


def format_scalar(_field_name: str, value: mt.Scalar | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float | np.floating):
        return f"{value:.3e}"
    return str(value)


class ReprMixin(AttrsBase):
    def __repr__(self) -> str:
        if self.association == pv.FieldAssociation.NONE:
            return self._format_items()
        table: prettytable.PrettyTable = self._describe_table()
        return table.get_string()

    def _format_items(self) -> str:
        result: str = ""
        max_len: int = max(map(len, self.keys()))
        for k, v in self.items():
            value = v.item() if v.size == 1 else v
            result += f"{k:{max_len}} : {value}\n"
        return result

    def _describe_table(self) -> prettytable.PrettyTable:
        table = prettytable.PrettyTable(
            [
                "Name",
                "Field",
                "Type",
                "N Comp",
                "Count",
                "Unique",
                "Top",
                "Freq",
                "Mean",
                "Min",
                "25%",
                "50%",
                "75%",
                "Max",
            ],
        )
        table.set_style(prettytable.MSWORD_FRIENDLY)
        for k in ["Top", "Mean", "Min", "25%", "50%", "75%", "Max"]:
            table.custom_format[k] = format_scalar
        for k, v in self.items():
            result: DescribeResult = describe_array(v)
            table.add_row(
                [
                    k,
                    self.association.name,
                    v.dtype,
                    v.shape[1] if v.ndim > 1 else "SCALAR",
                    result.count,
                    result.unique if result.unique < len(v) else "",
                    result.top if result.freq > 1 else "",
                    result.freq if result.freq > 1 else "",
                    result.mean,
                    result.min,
                    result.p25,
                    result.p50,
                    result.p75,
                    result.max,
                ]
            )
        return table
