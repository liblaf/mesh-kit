from typing import TYPE_CHECKING, TypeVar

import mkit.io as mi
import mkit.math as mm
import mkit.typing.numpy as tn

if TYPE_CHECKING:
    import pyvista as pv

_T = TypeVar("_T")


def transform(
    mesh: _T,
    transformation: tn.F44Like | None = None,
    *,
    transform_all_input_vectors: bool = False,
) -> _T:
    if transformation is None:
        return mesh
    mesh_pv: pv.PolyData = mi.pyvista.as_poly_data(mesh)
    mesh_pv = mesh_pv.transform(
        mm.as_numpy(transformation),
        transform_all_input_vectors=transform_all_input_vectors,
        inplace=False,
    )  # pyright: ignore [reportAssignmentType]
    return mi.convert(mesh_pv, type(mesh))
