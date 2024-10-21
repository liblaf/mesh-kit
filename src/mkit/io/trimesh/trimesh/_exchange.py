from __future__ import annotations

from typing import Any

import pyvista as pv
import trimesh as tm

import mkit.io.exchange as mie
import mkit.typing as mt


def as_trimesh(obj: Any) -> tm.Trimesh:
    return mie.convert(obj, tm.Trimesh)


class PyvistaPolyDataToTrimeshTrimesh(mie.ConverterBase):
    _to: type = tm.Trimesh

    def match_from(self, from_: Any) -> bool:
        return mt.is_instance_named_partial(from_, mie.ClassName.PYVISTA_POLY_DATA)

    def convert(
        self,
        from_: pv.PolyData,
        *,
        point_data: mt.AttrsLike | None = None,
        cell_data: mt.AttrsLike | None = None,
        field_data: mt.AttrsLike | None = None,
    ) -> tm.Trimesh:
        mesh: pv.PolyData = from_.triangulate()  # pyright: ignore [reportAssignmentType]
        self.warn_not_supported_association(pv.FieldAssociation.POINT, point_data)
        self.warn_not_supported_association(pv.FieldAssociation.CELL, cell_data)
        self.warn_not_supported_association(pv.FieldAssociation.NONE, field_data)
        return tm.Trimesh(mesh.points, mesh.regular_faces)


mie.register(PyvistaPolyDataToTrimeshTrimesh())
