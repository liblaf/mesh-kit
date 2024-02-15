import functools
import pathlib
from collections.abc import Generator
from typing import Optional

import numpy as np
import trimesh
from loguru import logger
from numpy import typing as npt
from scipy import sparse
from scipy.sparse import linalg
from trimesh import util

from mesh_kit import tetgen
from mesh_kit.common import testing
from mesh_kit.io import record as _record
from mesh_kit.registration import config as _config
from mesh_kit.registration import correspondence
from mesh_kit.registration import utils as _utils
from mesh_kit.std import time as _time


def nricp_amberg(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    source_landmarks: npt.NDArray,
    target_positions: npt.NDArray,
    config: Optional[_config.Config] = None,
    record_dir: Optional[pathlib.Path] = None,
) -> None:
    if not config:
        config = _config.Config()

    centroid: npt.NDArray = source_mesh.centroid
    scale: float = source_mesh.scale
    _normalize = functools.partial(_utils.normalize, centroid=centroid, scale=scale)
    _denormalize = functools.partial(_utils.denormalize, centroid=centroid, scale=scale)
    num_vertices: int = source_mesh.vertices.shape[0]
    num_edges: int = source_mesh.edges.shape[0]
    source_mesh = _normalize(source_mesh)
    target_mesh = _normalize(target_mesh)
    target_positions = _normalize(target_positions)
    if record_dir is not None:
        target_mesh.export(record_dir / "target.ply")
        np.savetxt(record_dir / "source-landmarks.txt", source_landmarks)
        np.savetxt(record_dir / "target-positions.txt", target_positions)

    last_params: Optional[_config.Params] = None
    for i, params in enumerate(config.steps):
        if i == 0 or params.rebase:
            # Unknowns 4x3 transformations X (Eq. 1)
            X: npt.NDArray = _create_X(num_vertices)
            testing.assert_shape(X.shape, (num_vertices * 4, 3))
            # D (Eq. 8)
            D: sparse.csr_matrix = _create_D(source_mesh.vertices)
            testing.assert_shape(D.shape, (num_vertices, num_vertices * 4))
            # D but for normal computation from the transformations X
            DN: sparse.csr_matrix = _create_D(source_mesh.vertex_normals)
            testing.assert_shape(DN.shape, (num_vertices, num_vertices * 4))
            # Node-arc incidence (M in Eq. 10)
            M: sparse.coo_matrix = _node_arc_incidence(source_mesh)
            testing.assert_shape(M.shape, (num_edges, num_vertices))
            # G (Eq. 10)
            G: npt.NDArray = np.diag([1.0, 1.0, 1.0, config.gamma])
            testing.assert_shape(G.shape, (4, 4))
            # M kronecker G (Eq. 10)
            M_kron_G: sparse.coo_matrix = sparse.kron(M, G)
            testing.assert_shape(M_kron_G.shape, (num_edges * 4, num_vertices * 4))
            Dl: sparse.csr_matrix
            Ul: npt.NDArray
            Dl, Ul = _create_Dl_Ul(D, source_mesh, source_landmarks, target_positions)
            testing.assert_shape(
                Dl.shape, (source_landmarks.shape[0], num_vertices * 4)
            )
            testing.assert_shape(Ul.shape, (source_landmarks.shape[0], 3))

        if config.watertight:
            current_params: _config.Params = params.model_copy(deep=True)
            while True:
                Xs: list[npt.NDArray] = list(
                    _nricp_amber(
                        source_mesh=source_mesh,
                        target_mesh=target_mesh,
                        D=D,
                        Dl=Dl,
                        DN=DN,
                        M_kron_G=M_kron_G,
                        params=current_params,
                        Ul=Ul,
                        X=X,
                    )
                )
                if tetgen.check(source_mesh):
                    logger.success(current_params)
                    for X in Xs:
                        source_mesh.vertices = D * X
                        _record.save(
                            source_mesh, dir=record_dir, id="", params=current_params
                        )
                    last_params = current_params.model_copy(deep=True)
                    if current_params == params:
                        break
                    current_params = params.model_copy(deep=True)
                else:
                    logger.error(current_params)
                    if (
                        last_params is None
                        or current_params.weight.stiff
                        >= last_params.weight.stiff - params.eps
                    ):
                        current_params.weight.stiff *= 1.5
                    else:
                        current_params = (current_params + last_params) / 2
        else:
            X = list(
                _nricp_amber(
                    source_mesh=source_mesh,
                    target_mesh=target_mesh,
                    D=D,
                    Dl=Dl,
                    DN=DN,
                    M_kron_G=M_kron_G,
                    params=params,
                    Ul=Ul,
                    X=X,
                    record_dir=record_dir,
                )
            )[-1]
            source_mesh.vertices = D * X
    return _denormalize(source_mesh)


def _nricp_amber(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    *,
    D: sparse.csr_matrix,
    Dl: sparse.csr_matrix,
    DN: sparse.csr_matrix,
    M_kron_G: sparse.coo_matrix,
    params: _config.Params,
    Ul: npt.NDArray,
    X: npt.NDArray,
    record_dir: Optional[pathlib.Path] = None,
) -> Generator[npt.NDArray]:
    logger.debug(params)
    source_mesh = source_mesh.copy()
    num_vertices: int = source_mesh.vertices.shape[0]

    last_error: float = np.inf
    error: float = np.inf
    iter: int = 0
    while not (
        np.isfinite(last_error - error) and (last_error - error) < params.eps
    ) and (params.max_iter is None or iter < params.max_iter):
        _record.save(source_mesh, record_dir, id="", params=params)
        yield X
        distance: npt.NDArray
        nearest: npt.NDArray
        target_normals: npt.NDArray
        distance, nearest, target_normals = correspondence.correspondence(
            source_mesh, target_mesh, config=params.correspondence
        )
        testing.assert_shape(distance.shape, (num_vertices,))
        testing.assert_shape(nearest.shape, (num_vertices, 3))
        testing.assert_shape(target_normals.shape, (num_vertices, 3))

        # Data weighting
        vertices_weight: npt.NDArray = np.ones(num_vertices)
        vertices_weight[distance > params.correspondence.threshold] = 0.0
        # Normal weighting = multiplying weights by cosines^wn
        source_normals: npt.NDArray = DN * X
        testing.assert_shape(source_normals.shape, (num_vertices, 3))
        dot: npt.NDArray = util.diagonal_dot(source_normals, target_normals)
        testing.assert_shape(dot.shape, (num_vertices,))
        # Normal orientation is only known for meshes as target_mesh
        dot = np.clip(dot, 0.0, 1.0)
        vertices_weight *= dot**params.weight.normal

        # Actual system solve
        X = _solve_system(
            M_kron_G=M_kron_G,
            D=D,
            vertices_weight=vertices_weight,
            nearest=nearest,
            weight_stiff=params.weight.stiff,
            Dl=Dl,
            Ul=Ul,
            weight_landmark=params.weight.landmark,
        )
        source_mesh.vertices = D * X
        last_error = error
        error_vec: npt.NDArray = np.linalg.norm(nearest - source_mesh.vertices, axis=-1)
        testing.assert_shape(error_vec.shape, (num_vertices,))
        error = (error_vec * vertices_weight).mean()
        logger.info("Error: {}", error)
        iter += 1
    _record.save(source_mesh, record_dir, id="", params=params)
    yield X


def _create_X(num_vertices: int) -> npt.NDArray:
    """Create Unknowns Matrix (Eq. 1)"""
    X: npt.NDArray = np.concatenate((np.eye(3), np.zeros(shape=(1, 3))), axis=0)
    return np.tile(X, reps=(num_vertices, 1))


def _create_D(vertices: npt.NDArray) -> sparse.csr_matrix:
    """Create Data matrix (Eq. 8)"""
    num_vertices: int = vertices.shape[0]
    testing.assert_shape(vertices.shape, (num_vertices, 3))
    rows: npt.NDArray = np.repeat(np.arange(num_vertices), repeats=4)
    cols: npt.NDArray = np.arange(num_vertices * 4)
    data: npt.NDArray = np.concatenate(
        (vertices, np.ones((num_vertices, 1))), axis=-1
    ).flatten()
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(num_vertices, num_vertices * 4)
    )


def _node_arc_incidence(mesh: trimesh.Trimesh) -> sparse.coo_matrix:
    """Computes node-arc incidence matrix of mesh (Eq. 10)"""
    num_vertices: int = mesh.vertices.shape[0]
    num_edges: int = mesh.edges.shape[0]
    rows: npt.NDArray = np.repeat(np.arange(num_edges), repeats=2)
    cols: npt.NDArray = mesh.edges.flatten()
    data: npt.NDArray = np.ones(num_edges * 2, np.float32)
    data[1::2] = -1.0
    edge_lengths: npt.NDArray = np.linalg.norm(
        mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]], axis=-1
    )
    data *= np.repeat(1.0 / edge_lengths, repeats=2)
    return sparse.coo_matrix((data, (rows, cols)), shape=(num_edges, num_vertices))


def _create_Dl_Ul(
    D: sparse.csr_matrix,
    source_mesh: trimesh.Trimesh,
    source_landmarks: npt.NDArray,
    target_positions: npt.NDArray,
) -> tuple[sparse.csr_matrix, npt.NDArray]:
    """Create landmark terms (Eq. 11)"""
    num_vertices: int = source_mesh.vertices.shape[0]
    num_landmarks: int = source_landmarks.shape[0]
    testing.assert_shape(D.shape, (num_vertices, num_vertices * 4))
    testing.assert_shape(source_landmarks.shape, (num_landmarks,))
    testing.assert_shape(target_positions.shape, (num_landmarks, 3))
    Dl: sparse.csr_matrix = D[source_landmarks, :]
    testing.assert_shape(Dl.shape, (num_landmarks, num_vertices * 4))
    Ul = target_positions
    testing.assert_shape(Ul.shape, (num_landmarks, 3))
    return Dl, Ul


@_time.timeit
def _solve_system(
    M_kron_G: sparse.coo_matrix,
    D: sparse.csr_matrix,
    vertices_weight: npt.NDArray,
    nearest: npt.NDArray,
    weight_stiff: float,
    Dl: sparse.csr_matrix,
    Ul: npt.NDArray,
    weight_landmark: float,
) -> npt.NDArray:
    """Solve for Eq. 12"""
    num_vertices: int = M_kron_G.shape[1] // 4
    num_edges: int = M_kron_G.shape[0] // 4
    num_landmarks: int = Dl.shape[0]
    testing.assert_shape(M_kron_G.shape, (num_edges * 4, num_vertices * 4))
    testing.assert_shape(D.shape, (num_vertices, num_vertices * 4))
    testing.assert_shape(vertices_weight.shape, (num_vertices,))
    testing.assert_shape(nearest.shape, (num_vertices, 3))
    testing.assert_shape(Dl.shape, (num_landmarks, num_vertices * 4))
    testing.assert_shape(Ul.shape, (num_landmarks, 3))
    U: npt.NDArray = nearest * vertices_weight[:, None]
    testing.assert_shape(U.shape, (num_vertices, 3))
    A_stack: list = [
        weight_stiff * M_kron_G,
        D.multiply(vertices_weight[:, None]),
        weight_landmark * Dl,
    ]
    B_shape: tuple[int] = (4 * num_edges + num_vertices + num_landmarks, 3)
    A: sparse.csr_matrix = sparse.vstack(A_stack, format="csr")
    B: sparse.lil_matrix = sparse.lil_matrix(B_shape)
    B[4 * num_edges : 4 * num_edges + num_vertices, :] = U
    B[
        4 * num_edges + num_vertices : (4 * num_edges + num_vertices + num_landmarks), :
    ] = Ul * weight_landmark
    X: sparse.csc_matrix = linalg.spsolve(A.T * A, A.T * B)
    return X.toarray()
