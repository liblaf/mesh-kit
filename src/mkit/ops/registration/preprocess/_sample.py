from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv

import mkit
import mkit.typing.numpy as nt

if TYPE_CHECKING:
    import trimesh as tm


def downsample_mesh(
    mesh: Any, *, density: float = 1e4, agg: int = 7, verbose: bool = True
) -> pv.PolyData:
    import fast_simplification

    mesh: pv.PolyData = mkit.io.pyvista.as_poly_data(mesh)
    count: int = int(density * mesh.area / mesh.length**2) + 1
    if mesh.n_faces_strict <= count:
        return mesh
    mesh = fast_simplification.simplify_mesh(
        mesh, target_count=count, agg=agg, verbose=verbose
    )
    return mesh


def mask_points(
    mesh: Any, mask: nt.BNLike | None = None, *, progress_bar: bool = True
) -> pv.PolyData:
    mesh: pv.PolyData = mkit.io.pyvista.as_poly_data(mesh)
    if mask is None:
        return mesh
    mask: nt.BN = mkit.math.numpy.as_bool(mask)
    unstructured: pv.UnstructuredGrid = mesh.extract_points(
        mask, progress_bar=progress_bar
    )
    mesh = unstructured.extract_surface(progress_bar=progress_bar)
    return mesh


def sample_points(mesh: Any, density: float = 1e4) -> pv.PolyData:
    mesh: tm.Trimesh = mkit.io.trimesh.as_trimesh(mesh)
    count: int = int(density * mesh.area / mesh.scale**2) + 1
    samples: nt.FN3
    if len(mesh.vertices) < count / 2:
        samples = mesh.sample(count - len(mesh.vertices))
        samples = np.vstack([mesh.vertices, samples])
    else:
        samples = mesh.sample(count)
    pcd: pv.PolyData = pv.wrap(samples)
    return pcd
