import pathlib
from typing import Any, Optional, Sequence

import numpy as np
import pydantic
import tqdm
import trimesh
from numpy import typing as npt
from scipy import interpolate, sparse, spatial
from trimesh import util

from mesh_kit.common import testing


class Params(pydantic.BaseModel):
    weight_smooth: float
    weight_landmark: float
    weight_normal: float
    max_iter: int = 10
    eps: float = 1e-4
    distance_threshold: float = 0.1
    correspondence_scale: float = 1
    correspondence_weight_normal: float = 0.5


def params_interp(
    x: Sequence[float], xp: Sequence[Params], fp: Sequence[float]
) -> Sequence[Params]:
    return [
        Params(
            **dict(zip(Params.model_fields, y)),
        )
        for y in interpolate.interp1d(
            x=xp,
            y=[[*step.model_dump().values()] for step in fp],
            axis=0,
        )(x)
    ]


def nricp_amberg(
    source_mesh: trimesh.Trimesh,
    target_geometry: trimesh.Trimesh,
    source_landmarks: Optional[npt.NDArray] = None,
    target_positions: Optional[npt.NDArray] = None,
    steps: Optional[Sequence[Params]] = None,
    # eps: float = 1e-4,
    gamma: float = 1,
    # distance_threshold: float = 0.1,
    return_records: bool = False,
    use_faces: bool = True,
    use_vertex_normals: bool = True,
    neighbors_count: int = 8,
    *,
    record_dir: Optional[pathlib.Path] = None,
) -> npt.NDArray | Sequence[npt.NDArray]:
    """Non Rigid Iterative Closest Points

    Implementation of "Amberg et al. 2007: Optimal Step
    Nonrigid ICP Algorithms for Surface Registration."
    Alloweight_smooth to register non-rigidly a mesh on another or
    on a point cloud. The core algorithm is explained
    at the end of page 3 of the paper.

    Comparison between nricp_amberg and nricp_sumner:

    - nricp_amberg fits to the target mesh in less steps
    - nricp_amberg can generate sharp edges
      - only vertices and their neighbors are considered
    - nricp_sumner tend to preserve more the original shape
    - nricp_sumner parameters are easier to tune
    - nricp_sumner solves for triangle positions whereas
        nricp_amberg solves for vertex transforms
    - nricp_sumner is less optimized when wn > 0

    Args:
      source_mesh: Trimesh
        Source mesh containing both vertices and faces.
      target_mesh: Trimesh or PointCloud or (n, 3) float
        Target geometry. It can contain no faces or be a PointCloud.
      source_landmarks: (n,) int or ((n,) int, (n, 3) float)
        n landmarks on the the source mesh.
        Represented as vertex indices (n,) int.
        It can also be represented as a tuple of triangle
        indices and barycentric coordinates ((n,) int, (n, 3) float,).
      target_positions: (n, 3) float
        Target positions assigned to source landmarks
      steps: Core parameters of the algorithm
        Iterable of iterables (ws, wl, wn, max_iter,).
        ws is smoothness term, wl weights landmark importance, wn normal importance
        and max_iter is the maximum number of iterations per step.
      eps: float
        If the error decrease if inferior to this value, the current step ends.
      gamma: float
        Weight the translation part against the rotational/skew part.
        Recommended value : 1.
      distance_threshold: float
        Distance threshold to account for a vertex match or not.
      return_records: bool
        If True, also returns all the intermediate results. It can help debugging
        and tune the parameters to match a specific case.
      use_faces: bool
        If True and if target geometry has faces, use proximity.closest_point to find
        matching points. Else use scipy's cKDTree object.
      use_vertex_normals: bool
        If True and if target geometry has faces, interpolate the normals of the target
        geometry matching points.
        Else use face normals or estimated normals if target geometry has no faces.
      neighbors_count: int
        number of neighbors used for normal estimation. Only used if target geometry has
        no faces or if use_faces is False.

    Returns:
      result: (n, 3) float or List[(n, 3) float]
        The vertices positions of source_mesh such that it is registered non-rigidly
        onto the target geometry.
        If return_records is True, it returns the list of the vertex positions at each
        iteration.
    """
    centroid: npt.NDArray
    scale: float
    target_geometry, target_positions, centroid, scale = _normalize_by_source(
        source_mesh=source_mesh,
        target_geometry=target_geometry,
        target_positions=target_positions,
    )
    testing.assert_shape(centroid, (3,))

    # Number of edges and vertices in source mesh
    num_edges: int = source_mesh.edges.shape[0]
    num_vertices: int = source_mesh.vertices.shape[0]
    num_landmarks: int = 0 if source_landmarks is None else source_landmarks.shape[0]
    if num_landmarks > 0:
        testing.assert_shape(source_landmarks, (num_landmarks,))
        testing.assert_shape(target_positions, (num_landmarks, 3))

    # Initialize transformed vertices
    transformed_vertices: npt.NDArray = source_mesh.vertices.copy()
    testing.assert_shape(transformed_vertices, (num_vertices, 3))
    # Node-arc incidence (M in Eq. 10)
    M: sparse.coo_matrix = _node_arc_incidence(mesh=source_mesh, do_weight=True)
    testing.assert_shape(M, (num_edges, num_vertices))
    # G (Eq. 10)
    G: npt.NDArray = np.diag([1, 1, 1, gamma])
    testing.assert_shape(G, (4, 4))
    # M kronecker G (Eq. 10)
    M_kron_G: sparse.coo_matrix = sparse.kron(M, G)
    testing.assert_shape(M_kron_G, (4 * num_edges, 4 * num_vertices))
    # D (Eq. 8)
    D: sparse.csr_matrix = _create_D(source_mesh.vertices)
    testing.assert_shape(D, (num_vertices, 4 * num_vertices))
    # D but for normal computation from the transformations X
    DN: sparse.csr_matrix = _create_D(source_mesh.vertex_normals)
    testing.assert_shape(DN, (num_vertices, 4 * num_vertices))
    # Unknowns 4x3 transformations X (Eq. 1)
    X: npt.NDArray = _create_X(num_vertices)
    testing.assert_shape(X, (4 * num_vertices, 3))
    # Landmark related terms (Eq. 11)
    Dl: Optional[npt.NDArray]
    Ul: Optional[npt.NDArray]
    Dl, Ul = _create_Dl_Ul(D, source_mesh, source_landmarks, target_positions)
    if num_landmarks > 0:
        testing.assert_shape(Dl, (num_landmarks, 4 * num_vertices))
        testing.assert_shape(Ul, (num_landmarks, 3))

    # Parameters of the algorithm (Eq. 6)
    # order : Alpha, Beta, normal weighting, and max iteration for step
    if steps is None:
        steps = [
            Params(weight_smooth=0.01, weight_landmark=10, weight_normal=0.5),
            Params(weight_smooth=0.02, weight_landmark=5, weight_normal=0.5),
            Params(weight_smooth=0.03, weight_landmark=2.5, weight_normal=0.5),
            Params(weight_smooth=0.01, weight_landmark=0, weight_normal=0),
        ]
    if return_records:
        records: Sequence[npt.NDArray] = [transformed_vertices]

    # Main loop
    progress: tqdm.tqdm = tqdm.tqdm(
        desc="Register", total=sum(step.max_iter for step in steps)
    )
    if record_dir is not None:
        record_count: int = 0
    for iter, params in enumerate(steps):
        # If normals are estimated from points and if there are less
        # than 3 points per query, avoid normal estimation
        if not use_faces and neighbors_count < 3:
            params.weight_normal = 0

        last_error: float = np.inf
        error: float = np.inf
        cpt_iter: int = 0

        # Current step iterations loop
        while (
            np.isinf(error)
            or last_error - error > params.eps
            and (params.max_iter is None or cpt_iter < params.max_iter)
        ):
            qres: dict[str, npt.NDArray] = _from_mesh(
                mesh=target_geometry,
                D=D,
                DN=DN,
                X=X,
                from_vertices_only=not use_faces,
                return_normals=params.weight_normal > 0,
                return_interpolated_normals=params.weight_normal > 0
                and use_vertex_normals,
                neighbors_count=neighbors_count,
                distance_threshold=params.distance_threshold,
                correspondence_scale=params.correspondence_scale,
                correspondence_weight_normal=params.correspondence_weight_normal,
            )

            # Data weighting
            vertices_weight: npt.NDArray = np.ones(num_vertices)
            testing.assert_shape(vertices_weight, (num_vertices,))
            vertices_weight[qres["distances"] > params.distance_threshold] = 0

            if params.weight_normal > 0 and "normals" in qres:
                target_normals = qres["normals"]
                if use_vertex_normals and "interpolated_normals" in qres:
                    target_normals = qres["interpolated_normals"]
                testing.assert_shape(target_normals, (num_vertices, 3))
                # Normal weighting = multiplying weights by cosines^wn
                source_normals: npt.ArrayLike = DN * X
                testing.assert_shape(source_normals, (num_vertices, 3))
                dot: npt.NDArray = util.diagonal_dot(source_normals, target_normals)
                testing.assert_shape(dot, (num_vertices,))
                # Normal orientation is only known for meshes as target
                dot = np.clip(dot, 0, 1) if use_faces else np.abs(dot)
                vertices_weight = vertices_weight * dot**params.weight_normal

            if record_dir is not None:
                (record_dir / f"{record_count:03d}-params.json").write_text(
                    params.model_dump_json(indent=2)
                )
                trimesh.Trimesh(
                    vertices=scale * transformed_vertices + centroid[None, :],
                    faces=source_mesh.faces,
                ).export(record_dir / f"{record_count:03d}.ply")
                np.savetxt(
                    record_dir / f"{record_count:03d}-source-correspondence.txt",
                    scale
                    * transformed_vertices[
                        qres["distances"] <= params.distance_threshold
                    ]
                    + centroid[None, :],
                )
                np.savetxt(
                    record_dir / f"{record_count:03d}-target-correspondence.txt",
                    scale
                    * qres["nearest"][qres["distances"] <= params.distance_threshold]
                    + centroid[None, :],
                )
                record_count += 1

            # Actual system solve
            X = _solve_system(
                M_kron_G,
                D,
                vertices_weight,
                qres["nearest"],
                params.weight_smooth,
                num_edges,
                num_vertices,
                Dl,
                Ul,
                params.weight_landmark,
            )
            transformed_vertices = D * X
            last_error = error
            error_vec: npt.NDArray = np.linalg.norm(
                qres["nearest"] - transformed_vertices, axis=-1
            )
            testing.assert_shape(error_vec, (num_vertices,))
            error = (error_vec * vertices_weight).mean()
            if return_records:
                records.append(transformed_vertices)

            cpt_iter += 1
            progress.update(1)
        progress.update(params.max_iter - cpt_iter)

    if return_records:
        result: Sequence[npt.NDArray] = records
    else:
        result: npt.NDArray = transformed_vertices

    result = _denormalize_by_source(
        source_mesh=source_mesh,
        target_geometry=target_geometry,
        target_positions=target_positions,
        result=result,
        centroid=centroid,
        scale=scale,
    )
    return result


def _normalize_by_source(
    source_mesh: trimesh.Trimesh,
    target_geometry: trimesh.Trimesh,
    target_positions: Optional[npt.NDArray],
) -> tuple[trimesh.Trimesh, Optional[npt.NDArray], npt.NDArray, float]:
    """Utility function to put the source mesh in [-1, 1]^3 and transform target geometry accordingly"""
    centroid: npt.NDArray
    scale: float
    centroid, scale = source_mesh.centroid, source_mesh.scale
    source_mesh.vertices = (source_mesh.vertices - centroid[None, :]) / scale
    # Dont forget to also transform the target positions
    target_geometry.vertices = (target_geometry.vertices - centroid[None, :]) / scale
    if target_positions is not None:
        target_positions = (target_positions - centroid[None, :]) / scale
    return target_geometry, target_positions, centroid, scale


def _node_arc_incidence(
    mesh: trimesh.Trimesh, do_weight: bool = True
) -> sparse.coo_matrix:
    """Computes node-arc incidence matrix of mesh (Eq.10)"""
    num_vertices: int = mesh.vertices.shape[0]
    num_edges: int = mesh.edges.shape[0]
    rows: npt.NDArray = np.repeat(np.arange(num_edges), 2)
    testing.assert_shape(rows, (2 * num_edges,))
    cols: npt.NDArray = mesh.edges.flatten()
    testing.assert_shape(cols, (2 * num_edges,))
    data: npt.NDArray = np.ones(2 * num_edges, np.float32)
    testing.assert_shape(data, (2 * num_edges,))
    data[1::2] = -1
    if do_weight:
        edge_lengths: npt.NDArray = np.linalg.norm(
            mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]], axis=-1
        )
        testing.assert_shape(edge_lengths, (num_edges,))
        data *= np.repeat(1 / edge_lengths, 2)
    return sparse.coo_matrix((data, (rows, cols)), shape=(num_edges, num_vertices))


def _create_D(vertex_3d_data: npt.NDArray) -> sparse.csr_matrix:
    """Create Data matrix (Eq. 8)"""
    num_vertices: int = vertex_3d_data.shape[0]
    rows: npt.NDArray = np.repeat(np.arange(num_vertices), 4)
    testing.assert_shape(rows, (4 * num_vertices,))
    cols: npt.NDArray = np.arange(4 * num_vertices)
    testing.assert_shape(cols, (4 * num_vertices,))
    data: npt.NDArray = np.concatenate(
        (vertex_3d_data, np.ones((num_vertices, 1))), axis=-1
    ).flatten()
    testing.assert_shape(data, (4 * num_vertices,))
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(num_vertices, 4 * num_vertices)
    )


def _create_X(num_vertices: int) -> npt.NDArray:
    """Create Unknowns Matrix (Eq. 1)"""
    X: npt.NDArray = np.concatenate((np.eye(3), np.array([[0, 0, 0]])), axis=0)
    testing.assert_shape(X, (4, 3))
    return np.tile(X, (num_vertices, 1))


def _create_Dl_Ul(
    D: npt.NDArray,
    source_mesh: trimesh.Trimesh,
    source_landmarks: npt.NDArray,
    target_positions: npt.NDArray,
) -> tuple[None, None] | tuple[npt.NDArray, npt.NDArray]:
    """Create landmark terms (Eq. 11)"""
    if source_landmarks is None or target_positions is None:
        # If no landmarks are provided, return None for both
        return None, None

    Dl: npt.NDArray = D[source_landmarks, :]
    Ul: npt.NDArray = target_positions
    num_vertices: int = source_mesh.vertices.shape[0]
    num_landmarks: int = source_landmarks.shape[0]
    testing.assert_shape(Dl, (num_landmarks, 4 * num_vertices))
    testing.assert_shape(Ul, (num_landmarks, 3))
    return Dl, Ul


def _from_mesh(
    mesh: trimesh.Trimesh,
    D: npt.NDArray,
    DN: npt.NDArray,
    X: npt.NDArray,
    from_vertices_only: bool = False,
    return_barycentric_coordinates: bool = False,
    return_normals: bool = False,
    return_interpolated_normals: bool = False,
    neighbors_count: int = 10,
    *,
    distance_threshold: float = 0.1,
    correspondence_scale: float = 1,
    correspondence_weight_normal: float = 0.5,
    **kwargs,
) -> dict[str, npt.NDArray]:
    """Find the the closest points and associated attributes from a Trimesh.

    Args:
      mesh: Trimesh
        Trimesh from which the query is performed
      input_points: (m, 3) float
        Input query points
      from_vertices_only: bool
        If True, consider only the vertices and not the faces
      return_barycentric_coordinates: bool
        If True, return the barycentric coordinates
      return_normals: bool
        If True, compute the normals at each closest point
      return_interpolated_normals: bool
        If True, return the interpolated normal at each closest point
      neighbors_count: int
        The number of closest neighbors to query
      kwargs: dict
        Dict to accept other key word arguments (not used)

    Returns:
      qres: Dict
        Dictionary containing :
          - nearest points (m, 3) with key 'nearest'
          - distances to nearest point (m,) with key 'distances'
          - support triangle indices of the nearest points (m,) with key 'tids'
          - [optional] normals at nearest points (m,3) with key 'normals'
          - [optional] barycentric coordinates in support triangles (m,3) with key
              'barycentric_coordinates'
          - [optional] interpolated normals (m,3) with key 'interpolated_normals'
    """
    num_vertices: int = D.shape[0]
    testing.assert_shape(D, (num_vertices, 4 * num_vertices))
    testing.assert_shape(DN, (num_vertices, 4 * num_vertices))
    testing.assert_shape(X, (4 * num_vertices, 3))
    input_points: npt.NDArray = correspondence_scale * D * X
    testing.assert_shape(input_points, (num_vertices, 3))
    input_normals: npt.NDArray = DN * X
    input_normals = input_normals / np.linalg.norm(input_normals, axis=-1)[:, None]
    testing.assert_shape(input_normals, (num_vertices, 3))
    neighbors_count = min(neighbors_count, len(mesh.vertices))

    assert not from_vertices_only
    # Else if we consider faces, use proximity.closest_point
    qres: dict[str, Any] = {}
    kd_tree: spatial.KDTree = mesh.kdtree
    distances: npt.NDArray
    idx: npt.NDArray
    distances, idx = kd_tree.query(input_points, neighbors_count)
    testing.assert_shape(distances, (num_vertices, neighbors_count))
    testing.assert_shape(idx, (num_vertices, neighbors_count))
    target_normals: npt.NDArray = mesh.vertex_normals[idx]
    testing.assert_shape(target_normals, (num_vertices, neighbors_count, 3))
    distances += correspondence_weight_normal * (
        1 - np.einsum("ijk,ijk->ij", input_normals[:, None, :], target_normals)
    )
    idx = idx[np.arange(num_vertices), np.argmin(distances, axis=-1)]
    testing.assert_shape(idx, (num_vertices,))
    qres["distances"] = np.min(distances, axis=-1)
    qres["nearest"] = mesh.vertices[idx]

    if return_normals:
        qres["normals"] = mesh.vertex_normals[idx]
        testing.assert_shape(qres["normals"], (num_vertices, 3))
        qres["interpolated_normals"] = qres["normals"]
    return qres

    assert not from_vertices_only
    # Else if we consider faces, use proximity.closest_point
    qres: dict[str, Any] = {
        "nearest": np.nan * np.zeros(shape=(num_vertices, 3)),
        "distances": np.nan * np.zeros(shape=(num_vertices,)),
    }
    results: Sequence[Sequence[int]] = spatial.cKDTree(input_points).query_ball_point(
        mesh.kdtree, r=distance_threshold
    )
    idx: npt.NDArray = np.zeros(shape=(num_vertices,), dtype=int)
    for i, result in enumerate(results):
        if len(result) == 0:
            continue
        distances: npt.NDArray = np.linalg.norm(
            input_points[i] - mesh.vertices[result], axis=-1
        ) + correspondence_weight_normal * (
            1 - np.sum(input_normals[i, :] * mesh.vertex_normals[result], axis=-1)
        )
        testing.assert_shape(distances, (len(result),))
        idx[i] = result[np.argmin(distances)]
        qres["nearest"][i] = mesh.vertices[idx[i]]
        qres["distances"][i] = distances.min()

    if return_normals:
        qres["normals"] = mesh.vertex_normals[idx]
        testing.assert_shape(qres["normals"], (num_vertices, 3))
        qres["interpolated_normals"] = qres["normals"]
    return qres


def _solve_system(
    M_kron_G: sparse.coo_matrix,
    D: sparse.csr_matrix,
    vertices_weight: npt.NDArray,
    nearest: npt.NDArray,
    weight_smooth: float,
    num_edges: int,
    num_vertices: int,
    Dl: Optional[npt.NDArray],
    Ul: Optional[npt.NDArray],
    weight_landmark: float,
) -> npt.NDArray:
    """Solve for Eq. 12"""
    testing.assert_shape(M_kron_G, (4 * num_edges, 4 * num_vertices))
    testing.assert_shape(D, (num_vertices, 4 * num_vertices))
    testing.assert_shape(vertices_weight, (num_vertices,))
    testing.assert_shape(nearest, (num_vertices, 3))
    U = nearest * vertices_weight[:, None]
    testing.assert_shape(U, (num_vertices, 3))
    use_landmarks: bool = Dl is not None and Ul is not None
    A_stack: Sequence[npt.ArrayLike] = [
        weight_smooth * M_kron_G,
        D.multiply(vertices_weight[:, None]),
    ]
    testing.assert_shape(A_stack[0], (4 * num_edges, 4 * num_vertices))
    testing.assert_shape(A_stack[1], (num_vertices, 4 * num_vertices))
    B_shape: tuple[int, int] = (4 * num_edges + num_vertices, 3)
    if use_landmarks:
        num_landmarks: int = Dl.shape[0]
        testing.assert_shape(Dl, (num_landmarks, 4 * num_vertices))
        testing.assert_shape(Ul, (num_landmarks, 3))
        A_stack.append(weight_landmark * Dl)
        testing.assert_shape(A_stack[-1], (num_landmarks, 4 * num_vertices))
        B_shape = (4 * num_edges + num_vertices + Ul.shape[0], 3)
    A = sparse.csr_matrix(sparse.vstack(A_stack))
    testing.assert_shape(
        A, (4 * num_edges + num_vertices + num_landmarks, 4 * num_vertices)
    )
    B = sparse.lil_matrix(B_shape, dtype=np.float32)
    testing.assert_shape(B, B_shape)
    B[4 * num_edges : (4 * num_edges + num_vertices), :] = U
    if use_landmarks:
        B[
            4 * num_edges + num_vertices : (4 * num_edges + num_vertices + Ul.shape[0]),
            :,
        ] = Ul * weight_landmark
    X: npt.NDArray = sparse.linalg.spsolve(A.T * A, A.T * B).toarray()
    testing.assert_shape(X, (4 * num_vertices, 3))
    return X


def _denormalize_by_source(
    source_mesh: trimesh.Trimesh,
    target_geometry: trimesh.Trimesh,
    target_positions: npt.NDArray,
    result: npt.NDArray | Sequence[npt.NDArray],
    centroid: npt.NDArray,
    scale: float,
) -> npt.NDArray | Sequence[npt.NDArray]:
    """Utility function to transform source mesh from [-1, 1]^3 to its original transform and transform target geometry accordingly"""
    testing.assert_shape(centroid, (3,))
    source_mesh.vertices = scale * source_mesh.vertices + centroid[None, :]
    target_geometry.vertices = scale * target_geometry.vertices + centroid[None, :]
    if target_positions is not None:
        testing.assert_shape(target_positions, (-1, 3))
        target_positions = scale * target_positions + centroid[None, :]
    if isinstance(result, list):
        [testing.assert_shape(x, (source_mesh.vertices.shape[0], 3)) for x in result]
        result = [scale * x + centroid[None, :] for x in result]
    else:
        testing.assert_shape(result, (source_mesh.vertices.shape[0], 3))
        result = scale * result + centroid[None, :]
    return result
