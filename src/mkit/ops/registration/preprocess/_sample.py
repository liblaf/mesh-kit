from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv

import mkit
import mkit.typing.numpy as nt

if TYPE_CHECKING:
    import trimesh as tm


def sample_points(mesh: Any, density: float = 1e4) -> pv.PolyData:
    mesh: tm.Trimesh = mkit.io.trimesh.as_trimesh(mesh)
    count: int = int(density * mesh.area) + 1
    samples: nt.DN3
    if len(mesh.vertices) < count / 2:
        samples = mesh.sample(count - len(mesh.vertices))
        samples = np.vstack([mesh.vertices, samples])
    else:
        samples = mesh.sample(count)
    pcd: pv.PolyData = pv.wrap(samples)
    return pcd
