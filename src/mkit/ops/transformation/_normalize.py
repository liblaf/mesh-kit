from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import trimesh.transformations as tf

import mkit.io as mi
import mkit.typing.numpy as tn

if TYPE_CHECKING:
    import pyvista as pv


def normalization_transformation(mesh: Any) -> tn.F44:
    return np.linalg.inv(denormalization_transformation(mesh))


def denormalization_transformation(mesh: Any) -> tn.F44:
    mesh: pv.PolyData = mi.pyvista.as_poly_data(mesh)
    return tf.scale_and_translate(mesh.length, mesh.center)


def normalize(mesh: Any, *, transform_all_input_vectors: bool = False) -> pv.PolyData:
    mesh: pv.PolyData = mi.pyvista.as_poly_data(mesh)
    mesh = mesh.transform(
        normalization_transformation(mesh),
        transform_all_input_vectors=transform_all_input_vectors,
        inplace=False,
    )  # pyright: ignore [reportAssignmentType]
    return mesh


def denormalize(
    mesh: Any, reference: Any, *, transform_all_input_vectors: bool = False
) -> pv.PolyData:
    mesh: pv.PolyData = mi.pyvista.as_poly_data(mesh)
    mesh = mesh.transform(
        denormalization_transformation(reference),
        transform_all_input_vectors=transform_all_input_vectors,
        inplace=False,
    )  # pyright: ignore [reportAssignmentType]
    return mesh
