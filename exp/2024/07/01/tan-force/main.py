import igl
import meshio
import mkit.array.points
import mkit.io
import numpy as np
import trimesh
from numpy.typing import NDArray

tetra: meshio.Mesh = mkit.io.load_meshio(
    "/home/liblaf/Documents/data/target/120056/predict.vtu"
)
mandible: trimesh.Trimesh = mkit.io.load_trimesh(
    "/home/liblaf/Documents/data/target/120056/post/99-mandible.vtu"
)
faces: NDArray[np.integer] = igl.boundary_facets(tetra.get_cells_type("tetra"))  # pyright: ignore [reportAttributeAccessIssue]
force: NDArray[np.float64] = np.asarray(tetra.point_data["force"]) * 1e9
mandible_force: NDArray[np.float64] = force[
    mkit.array.points.position_to_index(tetra.points, mandible.vertices)
]
normal_force: NDArray[np.float64] = np.abs(
    trimesh.util.diagonal_dot(mandible_force, mandible.vertex_normals)
)
tan_force: NDArray[np.float64] = np.sqrt(
    trimesh.util.diagonal_dot(mandible_force, mandible_force) - normal_force**2
)
mkit.io.save(
    "mandible.vtu",
    mandible,
    point_data={
        "normal_force": normal_force,
        "tan_force": tan_force,
        "tan_normal": tan_force / normal_force,
    },
)

skull: trimesh.Trimesh = trimesh.Trimesh(tetra.points, faces).split()[1]
vert_area: NDArray[np.float64] = np.zeros((len(skull.vertices),))
vert_area[skull.faces[:, 0]] += skull.area_faces
vert_area[skull.faces[:, 1]] += skull.area_faces
vert_area[skull.faces[:, 2]] += skull.area_faces
vert_area /= 3
skull_force: NDArray[np.float64] = force[
    mkit.array.points.position_to_index(tetra.points, skull.vertices)
]
normal_force = np.abs(trimesh.util.diagonal_dot(skull_force, skull.vertex_normals))
tan_force = np.sqrt(
    trimesh.util.diagonal_dot(skull_force, skull_force) - normal_force**2
)
mkit.io.save(
    "skull.vtu",
    skull,
    point_data={
        "normal_force": normal_force / vert_area,
        "tan_force": tan_force / vert_area,
        "tan_normal": tan_force / normal_force,
    },
)
