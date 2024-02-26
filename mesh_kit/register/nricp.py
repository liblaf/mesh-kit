import dataclasses
import functools
import pathlib
from collections.abc import Generator

import numpy as np
import trimesh
from loguru import logger
from numpy import typing as npt
from scipy import sparse
from scipy.sparse import linalg
from trimesh import util

from mesh_kit import log as _log
from mesh_kit import tetgen as _tetgen
from mesh_kit.io import record as _record
from mesh_kit.io import trimesh as _io
from mesh_kit.register import config as _config
from mesh_kit.register import nearest as _nearest
from mesh_kit.register import utils as _utils
from mesh_kit.typing import check_shape as _check_shape


@dataclasses.dataclass(kw_only=True)
class Vars:
    X: npt.NDArray  # Unknowns 4x3 transformations X (Eq. 1)
    D: sparse.csr_matrix  # D (Eq. 8)
    DN: sparse.csr_matrix  # D but for normal computation from the transformations X
    M: sparse.coo_matrix  # Node-arc incidence (M in Eq. 10)
    G: npt.NDArray  # G (Eq. 10)
    M_kron_G: sparse.coo_matrix  # M kronecker G (Eq. 10)
    Dl: sparse.csr_matrix
    Ul: npt.NDArray


def _init(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    source_landmarks: npt.NDArray,
    target_positions: npt.NDArray,
    config: _config.Config,
) -> Vars:
    num_verts: int = source_mesh.vertices.shape[0]
    num_edges: int = source_mesh.edges.shape[0]
    num_landmarks: int = source_landmarks.shape[0]
    # Unknowns 4x3 transformations X (Eq. 1)
    X: npt.NDArray = _check_shape(_create_X(num_verts), (num_verts * 4, 3))
    # D (Eq. 8)
    D: sparse.csr_matrix = _check_shape(
        _create_D(source_mesh.vertices), (num_verts, num_verts * 4)
    )
    # D but for normal computation from the transformations X
    DN: sparse.csr_matrix = _check_shape(
        _create_D(source_mesh.vertex_normals), (num_verts, num_verts * 4)
    )
    # Node-arc incidence (M in Eq. 10)
    M: sparse.coo_matrix = _check_shape(
        _node_arc_incidence(source_mesh), (num_edges, num_verts)
    )
    # G (Eq. 10)
    G: npt.NDArray = _check_shape(np.diag([1.0, 1.0, 1.0, config.gamma]), (4, 4))
    # M kronecker G (Eq. 10)
    M_kron_G: sparse.coo_matrix = _check_shape(
        sparse.kron(M, G), (num_edges * 4, num_verts * 4)
    )
    Dl: sparse.csr_matrix
    Ul: npt.NDArray
    Dl, Ul = _create_Dl_Ul(D, source_mesh, source_landmarks, target_positions)
    Dl = _check_shape(Dl, (num_landmarks, num_verts * 4))
    Ul = _check_shape(Ul, (num_landmarks, 3))
    return Vars(X=X, D=D, DN=DN, M=M, G=G, M_kron_G=M_kron_G, Dl=Dl, Ul=Ul)


def nricp_amberg(  # noqa: PLR0913
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    source_attrs: dict[str, npt.NDArray],
    target_attrs: dict[str, npt.NDArray],
    config: _config.Config | None = None,
    *,
    record_dir: pathlib.Path | None = None,
) -> trimesh.Trimesh:
    if not config:
        config = _config.Config()
    centroid: npt.NDArray = source_mesh.centroid
    scale: int = source_mesh.scale
    normalize = functools.partial(_utils.normalize, centroid=centroid, scale=scale)
    denormalize = functools.partial(_utils.denormalize, centroid=centroid, scale=scale)
    source_mesh = normalize(source_mesh)
    target_mesh = normalize(target_mesh)
    source_landmarks: npt.NDArray = source_attrs["landmarks"]
    target_positions: npt.NDArray = target_mesh.vertices[target_attrs["landmarks"]]
    if record_dir:
        _io.write(record_dir / "source.ply", source_mesh, attr=True, **source_attrs)
        _io.write(record_dir / "target.ply", target_mesh, attr=True, **target_attrs)
    last_params: _config.Params | None = None
    x: Vars = _init(
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        source_landmarks=source_landmarks,
        target_positions=target_positions,
        config=config,
    )
    for params in config.steps:
        if params.rebase:
            x = _init(
                source_mesh=source_mesh,
                target_mesh=target_mesh,
                source_landmarks=source_landmarks,
                target_positions=target_positions,
                config=config,
            )
        if config.watertight:
            current_params: _config.Params = params.model_copy(deep=True)
            while True:
                X: npt.NDArray = list(
                    _nricp_amber(
                        source_mesh=source_mesh,
                        target_mesh=target_mesh,
                        source_attrs=source_attrs,
                        target_attrs=target_attrs,
                        x=x,
                        params=current_params,
                        record_dir=record_dir,
                    )
                )[-1]
                source_mesh.vertices = x.D * X
                if _tetgen.check(source_mesh):
                    logger.success(current_params)
                    x.X = X
                    source_mesh.vertices = x.D * x.X
                    last_params = current_params.model_copy(deep=True)
                    if current_params == params:
                        break
                    current_params = params.model_copy(deep=True)
                else:
                    logger.error(current_params)
                    source_mesh.vertices = x.D * x.X
                    if (
                        last_params is None
                        or current_params.weight.stiff
                        >= last_params.weight.stiff - params.eps
                    ):
                        current_params.weight.stiff *= 1.5
                    else:
                        current_params = (current_params + last_params) / 2
        else:
            x.X = list(
                _nricp_amber(
                    source_mesh=source_mesh,
                    target_mesh=target_mesh,
                    source_attrs=source_attrs,
                    target_attrs=target_attrs,
                    x=x,
                    params=params,
                    record_dir=record_dir,
                )
            )[-1]
            source_mesh.vertices = x.D * x.X
    return denormalize(source_mesh)


def _nricp_amber(  # noqa: PLR0913
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    source_attrs: dict[str, npt.NDArray],
    target_attrs: dict[str, npt.NDArray],
    x: Vars,
    params: _config.Params,
    *,
    record_dir: pathlib.Path | None = None,
) -> Generator[npt.NDArray, None, None]:
    logger.debug(params)
    source_mesh = source_mesh.copy()
    num_verts: int = source_mesh.vertices.shape[0]

    last_error: float = np.inf
    error: float = np.inf
    cur_iter: int = 0
    X: npt.NDArray = x.X
    while not (
        np.isfinite(last_error - error) and (last_error - error) < params.eps
    ) and (params.max_iter is None or cur_iter < params.max_iter):
        _record.write(source_mesh, record_dir, span="", params=params)
        yield X
        distance: npt.NDArray
        nearest: npt.NDArray
        target_normals: npt.NDArray
        distance, nearest, target_normals = _nearest.nearest(
            source_mesh, target_mesh, config=params.nearest
        )
        distance = _check_shape(distance, (num_verts,))
        nearest = _check_shape(nearest, (num_verts, 3))
        target_normals = _check_shape(target_normals, (num_verts, 3))

        # Data weighting
        vertices_weight: npt.NDArray = np.ones(num_verts)
        threshold: float | npt.NDArray
        if "vert:distance-threshold" in source_attrs:
            logger.info("Using vert:distance-threshold")
            threshold = (
                params.nearest.threshold * source_attrs["vert:distance-threshold"]
            )
        else:
            logger.info("Using default threshold")
            threshold = params.nearest.threshold
        vertices_weight[distance > threshold] = 0.0
        # Normal weighting = multiplying weights by cosines^wn
        source_normals: npt.NDArray = _check_shape(x.DN * X, (num_verts, 3))
        dot: npt.NDArray = _check_shape(
            util.diagonal_dot(source_normals, target_normals), (num_verts,)
        )
        # Normal orientation is only known for meshes as target_mesh
        dot = np.clip(dot, 0.0, 1.0)
        vertices_weight *= dot**params.weight.normal

        # Actual system solve
        X = _solve_system(
            M_kron_G=x.M_kron_G,
            D=x.D,
            vertices_weight=vertices_weight,
            nearest=nearest,
            weight_stiff=params.weight.stiff,
            Dl=x.Dl,
            Ul=x.Ul,
            weight_landmark=params.weight.landmark,
        )
        source_mesh.vertices = x.D * X
        last_error = error
        error_vec: npt.NDArray = np.linalg.norm(nearest - source_mesh.vertices, axis=-1)
        error = (error_vec * vertices_weight).mean()
        logger.info("Error: {}", error)
        cur_iter += 1
    _record.write(source_mesh, record_dir, span="", params=params)
    yield X


def _create_X(num_verts: int) -> npt.NDArray:
    """Create Unknowns Matrix (Eq. 1)"""
    X: npt.NDArray = np.concatenate((np.eye(3), np.zeros(shape=(1, 3))), axis=0)
    return np.tile(X, reps=(num_verts, 1))


def _create_D(points: npt.NDArray) -> sparse.csr_matrix:
    """Create Data matrix (Eq. 8)"""
    num_verts: int = points.shape[0]
    points = _check_shape(points, (num_verts, 3))
    rows: npt.NDArray = np.repeat(np.arange(num_verts), repeats=4)
    cols: npt.NDArray = np.arange(num_verts * 4)
    data: npt.NDArray = np.concatenate(
        (points, np.ones((num_verts, 1))), axis=-1
    ).flatten()
    return sparse.csr_matrix((data, (rows, cols)), shape=(num_verts, num_verts * 4))


def _node_arc_incidence(mesh: trimesh.Trimesh) -> sparse.coo_matrix:
    """Computes node-arc incidence matrix of mesh (Eq. 10)"""
    num_verts: int = mesh.vertices.shape[0]
    num_edges: int = mesh.edges.shape[0]
    rows: npt.NDArray = np.repeat(np.arange(num_edges), repeats=2)
    cols: npt.NDArray = mesh.edges.flatten()
    data: npt.NDArray = np.ones(num_edges * 2, np.float32)
    data[1::2] = -1.0
    edge_lengths: npt.NDArray = np.linalg.norm(
        mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]], axis=-1
    )
    data *= np.repeat(1.0 / edge_lengths, repeats=2)
    return sparse.coo_matrix((data, (rows, cols)), shape=(num_edges, num_verts))


def _create_Dl_Ul(
    D: sparse.csr_matrix,
    source_mesh: trimesh.Trimesh,
    source_landmarks: npt.NDArray,
    target_positions: npt.NDArray,
) -> tuple[sparse.csr_matrix, npt.NDArray]:
    """Create landmark terms (Eq. 11)"""
    num_verts: int = source_mesh.vertices.shape[0]
    num_landmarks: int = source_landmarks.shape[0]
    D = _check_shape(D, (num_verts, num_verts * 4))
    source_landmarks = _check_shape(source_landmarks, (num_landmarks,))
    target_positions = _check_shape(target_positions, (num_landmarks, 3))
    Dl: sparse.csr_matrix = _check_shape(
        D[source_landmarks, :], (num_landmarks, num_verts * 4)
    )
    Ul: npt.NDArray = _check_shape(target_positions, (num_landmarks, 3))
    return Dl, Ul


@_log.timeit
def _solve_system(  # noqa: PLR0913
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
    num_verts: int = M_kron_G.shape[1] // 4
    num_edges: int = M_kron_G.shape[0] // 4
    num_landmarks: int = Dl.shape[0]
    M_kron_G = _check_shape(M_kron_G, (num_edges * 4, num_verts * 4))
    D = _check_shape(D, (num_verts, num_verts * 4))
    vertices_weight = _check_shape(vertices_weight, (num_verts,))
    nearest = _check_shape(nearest, (num_verts, 3))
    Dl = _check_shape(Dl, (num_landmarks, num_verts * 4))
    Ul = _check_shape(Ul, (num_landmarks, 3))
    U: npt.NDArray = _check_shape(nearest * vertices_weight[:, None], (num_verts, 3))
    A_stack: list = [
        weight_stiff * M_kron_G,
        D.multiply(vertices_weight[:, None]),
        weight_landmark * Dl,
    ]
    B_shape: tuple[int, int] = (4 * num_edges + num_verts + num_landmarks, 3)
    A: sparse.csr_matrix = sparse.vstack(A_stack, format="csr")
    B: sparse.lil_matrix = sparse.lil_matrix(B_shape)
    B[4 * num_edges : 4 * num_edges + num_verts, :] = U
    B[4 * num_edges + num_verts : (4 * num_edges + num_verts + num_landmarks), :] = (
        Ul * weight_landmark
    )
    X: sparse.csc_matrix = linalg.spsolve(A.T * A, A.T * B)
    return X.toarray()
