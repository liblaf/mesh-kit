from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import trimesh.transformations as tf

import mkit.io as mi
import mkit.typing.numpy as tn

if TYPE_CHECKING:
    import pyvista as pv

_T = TypeVar("_T")


def normalization_transformation(mesh: Any) -> tn.F44:
    return np.linalg.inv(denormalization_transformation(mesh))


def denormalization_transformation(mesh: Any) -> tn.F44:
    mesh: pv.PolyData = mi.pyvista.as_poly_data(mesh)
    return tf.scale_and_translate(mesh.length, mesh.center)


def normalize(mesh: _T, *, transform_all_input_vectors: bool = False) -> _T:
    mesh_pv: pv.PolyData = mi.pyvista.as_poly_data(mesh)
    mesh_pv = mesh_pv.transform(
        normalization_transformation(mesh),
        transform_all_input_vectors=transform_all_input_vectors,
        inplace=False,
    )  # pyright: ignore [reportAssignmentType]
    return mi.convert(mesh_pv, type(mesh))


def denormalize(
    mesh: _T, reference: Any, *, transform_all_input_vectors: bool = False
) -> _T:
    mesh_pv: pv.PolyData = mi.pyvista.as_poly_data(mesh)
    mesh_pv = mesh_pv.transform(
        denormalization_transformation(reference),
        transform_all_input_vectors=transform_all_input_vectors,
        inplace=False,
    )  # pyright: ignore [reportAssignmentType]
    return mi.convert(mesh_pv, type(mesh))
