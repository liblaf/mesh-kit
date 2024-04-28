import meshio
import numpy as np
import trimesh
from numpy import typing as npt


def to_trimesh(mesh: meshio.Mesh) -> trimesh.Trimesh:
    match mesh:
        case meshio.Mesh():
            return meshio2trimesh(mesh)
        case _:
            raise ValueError(
                f"Unsupported conversion from {type(mesh)} to trimesh.Trimesh"
            )


def meshio2trimesh(mesh: meshio.Mesh) -> trimesh.Trimesh:
    vertices: npt.NDArray[np.float64] = mesh.points
    faces: npt.NDArray[np.integer] = mesh.get_cells_type("triangle")
    return trimesh.Trimesh(vertices, faces)
