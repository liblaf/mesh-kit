from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyvista as pv

import mkit.io.exchange as mie
import mkit.ops as mo
import mkit.typing as mt

if TYPE_CHECKING:
    import trimesh as tm


class TrimeshToPyvistaPolyData(mie.ConverterBase):
    def match_from(self, from_: Any) -> bool:
        return mt.is_instance_named_partial(from_, mie.ClassName.TRIMESH)

    def match_to(self, to: type) -> bool:
        return mt.is_class_named_partial(to, mie.ClassName.PYVISTA_POLY_DATA)

    def convert(
        self,
        from_: tm.Trimesh,
        to: type[pv.PolyData],
        point_data: mt.AttrsLike | None = None,
        cell_data: mt.AttrsLike | None = None,
        field_data: mt.AttrsLike | None = None,
    ) -> pv.PolyData:
        mesh: pv.PolyData = pv.wrap(from_)  # pyright: ignore [reportAssignmentType]
        mo.update_attrs(mesh.point_data, point_data)
        mo.update_attrs(mesh.cell_data, cell_data)
        mo.update_attrs(mesh.field_data, field_data)
        return mesh
