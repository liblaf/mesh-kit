import functools
import logging
import pathlib
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pydantic
import trimesh
from numpy import typing as npt
from scipy import sparse, spatial
from scipy.sparse import linalg as sparse_linalg
from trimesh import util

from mesh_kit.common import testing
from mesh_kit.io import record


class Params(pydantic.BaseModel):
    class Weight(pydantic.BaseModel):
        stiffness: float
        landmark: float
        normal: float

    weight: Weight
    distance_upper_bound: float = 0.1
    eps: float = 1e-4
    max_iter: int = 10
    num_neighbors: int = 10
    workers: int = 1


def normalize(
    position: npt.NDArray, centroid: npt.NDArray, scale: float
) -> npt.NDArray:
    testing.assert_shape(centroid.shape, (3,))
    return (position - centroid) / scale


def normalize_mesh(
    mesh: trimesh.Trimesh, centroid: npt.NDArray, scale: float
) -> trimesh.Trimesh:
    mesh = mesh.copy()
    testing.assert_shape(centroid.shape, (3,))
    mesh.apply_translation(-centroid)
    mesh.apply_scale(1.0 / scale)
    return mesh


def denormalize(
    position: npt.NDArray, centroid: npt.NDArray, scale: float
) -> npt.NDArray:
    testing.assert_shape(centroid.shape, (3,))
    return position * scale + centroid


def denormalize_mesh(
    mesh: trimesh.Trimesh, centroid: npt.NDArray, scale: float
) -> trimesh.Trimesh:
    testing.assert_shape(centroid.shape, (3,))
    mesh = mesh.copy()
    mesh.apply_scale(scale)
    mesh.apply_translation(centroid)
    return mesh


def nricp_amber(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    source_landmarks: npt.NDArray,
    target_positions: npt.NDArray,
    *,
    steps: Optional[Sequence[Params]] = None,
    gamma: float = 1.0,
    record_dir: Optional[pathlib.Path] = None,
) -> trimesh.Trimesh:
    num_vertices: int = source_mesh.vertices.shape[0]
    num_edges: int = source_mesh.edges.shape[0]
    num_landmarks: int = source_landmarks.shape[0]

    centroid: npt.NDArray = source_mesh.centroid
    scale: float = source_mesh.scale
    _normalize = functools.partial(normalize, centroid=centroid, scale=scale)
    _denormalize = functools.partial(denormalize, centroid=centroid, scale=scale)
    _normalize_mesh = functools.partial(normalize_mesh, centroid=centroid, scale=scale)
    _denormalize_mesh = functools.partial(
        denormalize_mesh, centroid=centroid, scale=scale
    )
    source_mesh.vertices = _normalize(source_mesh.vertices)
    target_mesh.vertices = _normalize(target_mesh.vertices)

    # Node-arc incidence (M in Eq. 10)
    M: sparse.coo_matrix = node_arc_incidence(source_mesh)
    testing.assert_shape(M.shape, (num_edges, num_vertices))
    # G (Eq. 10)
    G: npt.NDArray = np.diag([1.0, 1.0, 1.0, gamma])
    # M kronecker G (Eq. 10)
    M_kron_G: sparse.coo_matrix = sparse.kron(M, G)
    testing.assert_shape(M_kron_G.shape, (4 * num_edges, 4 * num_vertices))
    # D (Eq. 8)
    D: sparse.csr_matrix = create_D(source_mesh.vertices)
    testing.assert_shape(D.shape, (num_vertices, 4 * num_vertices))
    # D but for normal computation from the transformations X
    DN: sparse.csr_matrix = create_D(source_mesh.vertex_normals)
    testing.assert_shape(DN.shape, (num_vertices, 4 * num_vertices))
    # Unknowns 4x3 transformations X (Eq. 1)
    X: npt.NDArray = create_X(num_vertices)
    testing.assert_shape(X.shape, (4 * num_vertices, 3))
    # Landmark related terms (Eq. 11)
    Dl: npt.NDArray
    Ul: npt.NDArray
    Dl, Ul = create_Dl_Ul(
        D=D,
        mesh=source_mesh,
        source_landmarks=source_landmarks,
        target_positions=target_positions,
    )
    testing.assert_shape(Dl.shape, (num_landmarks, 4 * num_vertices))
    testing.assert_shape(Ul.shape, (num_landmarks, 3))

    if steps is None:
        steps = [
            Params(weight=Params.Weight(stiffness=0.01, landmark=10.0, normal=0.5)),
            Params(weight=Params.Weight(stiffness=0.02, landmark=5.0, normal=0.5)),
            Params(weight=Params.Weight(stiffness=0.03, landmark=2.5, normal=0.5)),
            Params(weight=Params.Weight(stiffness=0.01, landmark=0.0, normal=0.0)),
        ]

    for params in steps:
        last_error: float = np.inf
        error: float = np.inf
        iter: int = 0
        while (
            np.isnan(last_error - error) or last_error - error > params.eps
        ) and iter < params.max_iter:
            target_idx: npt.NDArray = correspondence(
                source_mesh=source_mesh,
                target_mesh=target_mesh,
                num_neighbors=params.num_neighbors,
                distance_upper_bound=params.distance_upper_bound,
                workers=params.workers,
            )
            if record_dir is not None:
                record.save(
                    _denormalize_mesh(source_mesh),
                    dir=record_dir,
                    params=params,
                    source_positions=_denormalize(
                        source_mesh.vertices[target_idx >= 0]
                    ),
                    target_positions=_denormalize(
                        target_mesh.vertices[target_idx[target_idx >= 0]]
                    ),
                )
            testing.assert_shape(target_idx.shape, (num_vertices,))
            vertices_weight: npt.NDArray = np.ones((num_vertices,))
            target_normal: npt.NDArray = target_mesh.vertex_normals[target_idx]
            testing.assert_shape(target_normal.shape, (num_vertices, 3))
            source_normal: npt.NDArray = DN @ X
            testing.assert_shape(source_normal.shape, (num_vertices, 3))
            weight_normal: npt.NDArray = np.clip(
                util.diagonal_dot(source_normal, target_normal), a_min=0.0, a_max=1.0
            )
            testing.assert_shape(weight_normal.shape, (num_vertices,))
            vertices_weight = (1.0 - params.weight.normal) * vertices_weight + (
                params.weight.normal * weight_normal
            )
            vertices_weight[target_idx < 0] = 0.0
            print(vertices_weight)
            X = solve_system(
                M_kron_G=M_kron_G,
                D=D,
                vertices_weight=vertices_weight,
                nearest=target_mesh.vertices[target_idx],
                weight_stiffness=params.weight.stiffness,
                num_edges=num_edges,
                num_vertices=num_vertices,
                Dl=Dl,
                Ul=Ul,
                weight_landmark=params.weight.landmark,
            )
            testing.assert_shape(X.shape, (4 * num_vertices, 3))
            source_mesh.vertices = D * X

            last_error = error
            error = np.mean(
                vertices_weight
                * util.row_norm(target_mesh.vertices[target_idx] - source_mesh.vertices)
            )
            logging.info("Iter %d: error = %f", iter, error)
            iter += 1

    return _denormalize_mesh(source_mesh)


def solve_system(
    M_kron_G: sparse.coo_matrix,
    D: sparse.csr_matrix,
    vertices_weight: npt.NDArray,
    nearest: npt.NDArray,
    weight_stiffness: float,
    num_edges: int,
    num_vertices: int,
    Dl: npt.NDArray,
    Ul: npt.NDArray,
    weight_landmark: float,
) -> npt.NDArray:
    """Solve for Eq. 12"""
    num_landmarks: int = Dl.shape[0]
    testing.assert_shape(M_kron_G.shape, (4 * num_edges, 4 * num_vertices))
    testing.assert_shape(D.shape, (num_vertices, 4 * num_vertices))
    testing.assert_shape(vertices_weight.shape, (num_vertices,))
    testing.assert_shape(nearest.shape, (num_vertices, 3))
    testing.assert_shape(Dl.shape, (num_landmarks, 4 * num_vertices))
    testing.assert_shape(Ul.shape, (num_landmarks, 3))
    U: npt.NDArray = nearest * vertices_weight[:, None]
    A_stack: Sequence[npt.NDArray] = [
        weight_stiffness * M_kron_G,
        D.multiply(vertices_weight[:, None]),
        weight_landmark * Dl,
    ]
    B_shape: Sequence[int] = (4 * num_edges + num_vertices + num_landmarks, 3)
    A: sparse.csr_matrix = sparse.csr_matrix(sparse.vstack(A_stack))
    testing.assert_shape(
        A.shape, (4 * num_edges + num_vertices + num_landmarks, 4 * num_vertices)
    )
    B: sparse.lil_matrix = sparse.lil_matrix(B_shape, dtype=float)
    testing.assert_shape(B.shape, B_shape)
    B[4 * num_edges : (4 * num_edges + num_vertices), :] = U
    B[4 * num_edges + num_vertices :, :] = weight_landmark * Ul
    X: sparse.csr_matrix = sparse_linalg.spsolve(A.T * A, A.T * B)
    testing.assert_shape(X.shape, (4 * num_vertices, 3))
    return X.toarray()


def node_arc_incidence(mesh: trimesh.Trimesh) -> sparse.coo_matrix:
    """Computes node-arc incidence matrix of mesh (Eq. 10)"""
    num_vertices: int = mesh.vertices.shape[0]
    num_edges: int = mesh.edges.shape[0]
    rows: npt.NDArray = np.repeat(np.arange(num_edges), repeats=2)
    cols: npt.NDArray = mesh.edges.flatten()
    data: npt.NDArray = np.ones(2 * num_edges, dtype=float)
    data[1::2] = -1.0
    edge_lengths: npt.NDArray = np.linalg.norm(
        mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]], axis=-1
    )
    data *= np.repeat(1.0 / edge_lengths, repeats=2)
    return sparse.coo_matrix((data, (rows, cols)), shape=(num_edges, num_vertices))


def create_D(vertex_data: npt.NDArray) -> sparse.csr_matrix:
    """Create Data matrix (Eq. 8)"""
    num_vertices: int = vertex_data.shape[0]
    rows: npt.NDArray = np.repeat(np.arange(num_vertices), repeats=4)
    cols: npt.NDArray = np.arange(4 * num_vertices)
    data: npt.NDArray = np.concatenate(
        (vertex_data, np.ones((num_vertices, 1))), axis=-1
    ).flatten()
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(num_vertices, 4 * num_vertices)
    )


def create_X(num_vertices: int) -> npt.NDArray:
    """Create Unknowns Matrix (Eq. 1)"""
    x: npt.NDArray = np.concatenate((np.eye(3), np.array([[0, 0, 0]])), axis=0)
    return np.tile(x, reps=(num_vertices, 1))


def create_Dl_Ul(
    D: sparse.csr_matrix,
    mesh: trimesh.Trimesh,
    source_landmarks: npt.NDArray,
    target_positions: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Create landmark terms (Eq. 11)"""
    Dl: npt.NDArray = D[source_landmarks, :]
    Ul: npt.NDArray = target_positions
    return Dl, Ul


def correspondence(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    *,
    distance_upper_bound: float = np.inf,
    num_neighbors: int = 10,
    weight_normal: float = 1.0,
    workers: int = 1,
) -> npt.NDArray:
    num_vertices: int = source_mesh.vertices.shape[0]
    tree: spatial.KDTree = target_mesh.kdtree
    distance: npt.NDArray
    index: npt.NDArray
    distance, index = tree.query(
        source_mesh.vertices,
        k=num_neighbors,
        distance_upper_bound=distance_upper_bound,
        workers=workers,
    )
    testing.assert_shape(distance.shape, (num_vertices, num_neighbors))
    testing.assert_shape(index.shape, (num_vertices, num_neighbors))
    source_normal: npt.NDArray = source_mesh.vertex_normals
    testing.assert_shape(source_normal.shape, (num_vertices, 3))
    index[index == target_mesh.vertices.shape[0]] = -1
    target_normal: npt.NDArray = target_mesh.vertex_normals[index]
    testing.assert_shape(target_normal.shape, (num_vertices, num_neighbors, 3))
    loss_normal: npt.NDArray = 1.0 - np.einsum(
        "ik,ijk->ij", source_normal, target_normal
    )
    testing.assert_shape(loss_normal.shape, (num_vertices, num_neighbors))
    loss: npt.NDArray = distance + weight_normal * loss_normal
    testing.assert_shape(loss.shape, (num_vertices, num_neighbors))
    index = index[np.arange(num_vertices), np.argmin(loss, axis=-1)]
    testing.assert_shape(index.shape, (num_vertices,))
    return index
