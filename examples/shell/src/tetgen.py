import meshio
import numpy as np
import scipy
import scipy.interpolate
import trimesh
from icecream import ic
from mkit import cli, tetgen
from mkit.typing import as_any as _a
from numpy import typing as npt


def main() -> None:
    pre_face: trimesh.Trimesh = _a(trimesh.load("pre-face.ply"))
    pre_skull: trimesh.Trimesh = _a(trimesh.load("pre-skull.ply"))
    post_skull: trimesh.Trimesh = _a(trimesh.load("post-skull.ply"))
    mesh_tr: trimesh.Trimesh = pre_face.difference(pre_skull)
    mesh_io = meshio.Mesh(
        mesh_tr.vertices,
        [("triangle", mesh_tr.faces)],
        field_data={"holes": np.asarray([[0.0, 0.0, 0.0]])},
    )
    tetra: meshio.Mesh = tetgen.tetgen(mesh_io)

    distance: npt.NDArray[np.float64]
    _, distance, _ = pre_skull.nearest.on_surface(tetra.points)
    point_mask: npt.NDArray[np.bool_] = distance < (1e-9 * pre_skull.scale)
    interpolator = scipy.interpolate.LinearNDInterpolator(
        pre_skull.vertices, post_skull.vertices - pre_skull.vertices
    )
    displacement: npt.NDArray[np.float64] = np.full_like(tetra.points, np.nan)
    displacement[point_mask] = interpolator(tetra.points[point_mask])
    tetra.points[point_mask] += displacement[point_mask]
    skull = trimesh.Trimesh(
        tetra.points, tetra.get_cells_type("triangle"), process=False
    )
    skull.update_vertices(point_mask)
    skull.process()
    ic(skull)
    skull.export("post-skull-interpolate.ply")


if __name__ == "__main__":
    cli.run(main)
