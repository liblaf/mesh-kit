from typing import Any

import pyvista as pv


def tetgen(surface: Any) -> pv.UnstructuredGrid:
    import meshpy.tet

    import mkit.io

    surface_mesh: pv.PolyData = mkit.io.as_polydata(surface)
    surface_mesh = surface_mesh.triangulate(progress_bar=True)
    surface_info = meshpy.tet.MeshInfo()
    surface_info.set_points(surface_mesh.points)
    surface_info.set_facets(surface_mesh.regular_faces)
    tetmesh_info: meshpy.tet.MeshInfo = meshpy.tet.build(surface_info, verbose=True)
    tetmesh: pv.UnstructuredGrid = mkit.io.unstructured_grid_tetmesh(
        tetmesh_info.points, tetmesh_info.elements
    )
    return tetmesh
