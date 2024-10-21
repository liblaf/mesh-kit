from typing import Any

import pyvista as pv

import mkit.io as mi
import mkit.math as mm
import mkit.typing.numpy as tn


def select_points(
    mesh: Any,
    selection: tn.BNLike | tn.INLike | None = None,
    *,
    adjacent_cells: bool = True,
    invert: bool = False,
) -> pv.PolyData:
    mesh: pv.PolyData = mi.pyvista.as_poly_data(mesh)
    if selection is None:
        return mesh
    if invert:
        selection = ~mm.as_numpy(selection)
    unstructured_grid: pv.UnstructuredGrid = mesh.extract_points(
        selection, adjacent_cells=adjacent_cells, include_cells=True
    )  # pyright: ignore [reportAssignmentType]
    mesh = unstructured_grid.extract_surface()  # pyright: ignore [reportAssignmentType]
    return mesh


def select_cells(
    mesh: Any, selection: tn.INLike | None = None, *, invert: bool = False
) -> pv.PolyData:
    mesh: pv.PolyData = mi.pyvista.as_poly_data(mesh)
    if selection is None:
        return mesh
    unstructured_grid: pv.UnstructuredGrid = mesh.extract_cells(
        selection, invert=invert
    )  # pyright: ignore [reportAssignmentType]
    mesh = unstructured_grid.extract_surface()  # pyright: ignore [reportAssignmentType]
    return mesh
