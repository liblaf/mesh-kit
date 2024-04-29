import functools
import pathlib

import numpy as np
import pydantic
import pytorch3d.structures
import torch
import trimesh
from numpy import typing as npt
from pytorch3d import loss, ops, structures
from scipy import spatial

from mkit import io as _io
from mkit.array import mask


def register(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    source_vert_mask: npt.NDArray[np.bool_] | None = None,
    target_vert_mask: npt.NDArray[np.bool_] | None = None,
    record_dir: pathlib.Path | None = None,
) -> None:
    scale: float = source.scale
    centroid: npt.NDArray[np.float64] = source.centroid
    _normalize = functools.partial(normalize, centroid=centroid, scale=scale)
    _denormalize = functools.partial(denormalize, centroid=centroid, scale=scale)
    source = _normalize(source)
    target = _normalize(target)
    source_vert_mask, target_vert_mask = mask_overlap(
        source,
        target,
        source_vert_mask=source_vert_mask,
        target_vert_mask=target_vert_mask,
        record_dir=record_dir,
    )

    source_t3 = structures.Meshes(
        [torch.as_tensor(source.vertices)], [torch.as_tensor(source.faces)]
    )
    target_t3 = structures.Meshes(
        [torch.as_tensor(target.vertices)], [torch.as_tensor(target.faces)]
    )
    register_torch(
        source_t3,
        target_t3,
        source_face_mask=torch.as_tensor(
            mask.vertex_to_face(source.faces, source_vert_mask)
        ),
        target_face_mask=torch.as_tensor(
            mask.vertex_to_face(target.faces, target_vert_mask)
        ),
    )
    raise NotImplementedError  # TODO


def register_torch(
    source: structures.Meshes,
    target: structures.Meshes,
    *,
    source_face_mask: torch.Tensor,
    target_face_mask: torch.Tensor,
) -> torch.Tensor:
    def create_X(n_verts: int) -> torch.Tensor:
        X_cell: npt.NDArray[np.float64] = np.concatenate((np.eye(3), np.zeros(3)))
        assert X_cell.shape == (3, 4)
        X_np: npt.NDArray[np.float64] = np.tile(X_cell, (n_verts, 1, 1))
        assert X_np.shape == (n_verts, 3, 4)
        return torch.tensor(X_np, requires_grad=True)

    def transform(mesh: structures.Meshes, X: torch.Tensor) -> structures.Meshes:
        verts: torch.Tensor = mesh.verts_packed()
        n_verts: int = verts.shape[0]
        assert verts.shape == (n_verts, 3)
        assert X.shape == (n_verts, 3, 4)
        verts = torch.hstack((verts, torch.ones(n_verts, 1)))
        assert verts.shape == (n_verts, 4)
        verts = verts.reshape(n_verts, 4, 1)
        verts = torch.bmm(X, verts)
        return structures.Meshes([verts], mesh.faces_list())

    source_face_idx: torch.Tensor
    (source_face_idx,) = source_face_mask.nonzero(as_tuple=False)
    target_face_idx: torch.Tensor
    (target_face_idx,) = target_face_mask.nonzero(as_tuple=False)

    n_verts: int = source.verts_packed().shape[0]
    X: torch.Tensor = create_X(n_verts)
    optimizer = torch.optim.SGD([X], lr=1.0, momentum=0.9)
    for i in range(1):
        optimizer.zero_grad()
        result: structures.Meshes = transform(source, X)
        result_overlap: torch.Tensor = result.submeshes([source_face_idx])
        target_overlap: torch.Tensor = target.submeshes([target_face_idx])
        result_samples: torch.Tensor = ops.sample_points_from_meshes(result_overlap)
        target_samples: torch.Tensor = ops.sample_points_from_meshes(target_overlap)
        loss_chamfer: torch.Tensor = loss.chamfer_distance(
            result_samples, target_samples
        )
        loss_stiffness: torch.Tensor = _loss_stiffness(
            X, result.edges_packed(), G=torch.as_tensor([1.0, 1.0, 1.0, 1.0])
        )
        loss_edge: torch.Tensor = loss.mesh_edge_loss(result)
        loss_laplacian_smoothing: torch.Tensor = loss.mesh_laplacian_smoothing(
            result, "uniform"
        )
        loss_normal_consistency: torch.Tensor = loss.mesh_normal_consistency(result)
        loss_total: torch.Tensor = (
            loss_chamfer
            + loss_stiffness
            + loss_edge
            + loss_laplacian_smoothing
            + loss_normal_consistency
        )
        loss_total.backward()
        optimizer.step()
    result = transform(source, X)
    return result


class HyperParams(pydantic.BaseModel):
    max_iter: int = 1000
    lr: float = 1.0
    momentum: float = 0.9
    weight_distance: float = 1.0
    weight_stiffness: float = 1.0
    weight_edge: float = 0.0
    weight_laplacian_smoothing_uniform: float = 0.0
    weight_laplacian_smoothing_cot: float = 0.0  # TODO
    weight_laplacian_smoothing_cotcurv: float = 0.0  # TODO
    weight_normal_consistency: float = 0.0


def register_torch_inner(
    source: pytorch3d.structures.Meshes,
    target: pytorch3d.structures.Meshes,
    *,
    source_face_idx: torch.LongTensor,
    target_face_idx: torch.LongTensor,
    initial: torch.Tensor | None = None,
    iter_start: int = 0,
    max_iter: int = 1000,
) -> torch.Tensor:
    if initial:
        X: torch.Tensor = initial
    else:
        verts: torch.Tensor = source.verts_packed()
        n_verts: int = verts.shape[0]
        X = create_X(n_verts)
    optimizer = torch.optim.SGD([X], lr=1.0, momentum=0.9)
    for i in range(iter_start, iter_start + max_iter):
        optimizer.zero_grad()
        total_loss: torch.Tensor = torch.tensor(0.0, requires_grad=True)
        optimizer.step()
    raise NotImplementedError  # TODO


def apply_transform(mesh: structures.Meshes, X: torch.Tensor) -> structures.Meshes:
    verts: torch.Tensor = mesh.verts_packed()
    n_verts: int = verts.shape[0]
    assert verts.shape == (n_verts, 3)
    assert X.shape == (n_verts, 3, 4)
    verts = torch.hstack((verts, torch.ones(n_verts, 1)))
    assert verts.shape == (n_verts, 4)
    verts = verts.reshape(n_verts, 4, 1)
    verts = torch.bmm(X, verts)
    return structures.Meshes([verts], mesh.faces_list())


def compute_loss(
    X: torch.Tensor,
    source: pytorch3d.structures.Meshes,
    target: pytorch3d.structures.Meshes,
    *,
    source_face_idx: torch.LongTensor | None,
    target_face_idx: torch.LongTensor | None,
    params: HyperParams,
) -> torch.Tensor:
    total_loss: torch.Tensor = torch.tensor(0.0, requires_grad=True)
    result: pytorch3d.structures.Meshes = apply_transform(source, X)
    if params.weight_distance > 0:
        result_masked: pytorch3d.structures.Meshes = submesh(result, source_face_idx)
        target_masked: pytorch3d.structures.Meshes = submesh(target, target_face_idx)
        result_samples: torch.Tensor = ops.sample_points_from_meshes(result_masked)
        target_samples: torch.Tensor = ops.sample_points_from_meshes(target_masked)
        loss_distance: torch.Tensor = loss.chamfer_distance(
            result_samples, target_samples
        )
        total_loss += params.weight_distance * loss_distance
    if params.weight_stiffness > 0:
        edges: torch.LongTensor = source.edges_packed()
        loss_stiffness: torch.Tensor = _loss_stiffness(
            X, edges, torch.tensor([1.0, 1.0, 1.0, 1.0])
        )
        total_loss += params.weight_stiffness * loss_stiffness

    raise NotImplementedError


def submesh(
    mesh: pytorch3d.structures.Meshes, face_idx: torch.LongTensor | None = None
) -> pytorch3d.structures.Meshes:
    if face_idx is None:
        return mesh
    return mesh.submeshes([face_idx])


def create_X(n_verts: int) -> torch.Tensor:
    X_cell: npt.NDArray[np.float64] = np.concatenate((np.eye(3), np.zeros(3)))
    assert X_cell.shape == (3, 4)
    X_np: npt.NDArray[np.float64] = np.tile(X_cell, (n_verts, 1, 1))
    assert X_np.shape == (n_verts, 3, 4)
    return torch.tensor(X_np, requires_grad=True)


def _loss_stiffness(
    X: torch.Tensor, edges: torch.Tensor, G: torch.Tensor
) -> torch.Tensor:
    n_verts: int = X.shape[0]
    n_edges: int = edges.shape[0]
    assert X.shape == (n_verts, 3, 4)
    assert edges.shape == (n_edges, 2)

    def loss_per_edge(Xi: torch.Tensor, Xj: torch.Tensor) -> torch.Tensor:
        return torch.norm((Xi - Xj) @ G)

    assert G.shape == (4,)
    X_adj: torch.Tensor = X[edges[0]] - X[edges[1]]
    assert X_adj.shape == (n_edges, 3, 4)
    loss: torch.Tensor = torch.vmap(loss_per_edge)(X[edges[0]], X[edges[1]])
    assert loss.shape == (n_edges,)
    return loss.mean()


def mask_overlap(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    source_vert_mask: npt.NDArray[np.bool_] | None = None,
    target_vert_mask: npt.NDArray[np.bool_] | None = None,
    threshold: float = 0.05,
    record_dir: pathlib.Path | None = None,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    if source_vert_mask is None:
        source_vert_mask = np.ones(source.vertices.shape[0], np.bool_)
    if target_vert_mask is None:
        target_vert_mask = np.ones(target.vertices.shape[0], np.bool_)

    kdtree = spatial.KDTree(target.vertices[target_vert_mask])
    distance: npt.NDArray[np.float64]
    index: npt.NDArray[np.int64]
    distance, index = kdtree.query(
        source.vertices[source_vert_mask], distance_upper_bound=threshold
    )
    source_vert_mask[source_vert_mask] &= distance < threshold
    kdtree = spatial.KDTree(source.vertices[source_vert_mask])
    distance, index = kdtree.query(
        target.vertices[target_vert_mask], distance_upper_bound=threshold
    )
    target_vert_mask[target_vert_mask] &= distance < threshold

    if record_dir is not None:
        _io.save(
            record_dir / "source.vtk",
            source,
            point_data={"mask": source_vert_mask.astype(np.int8)},
        )
        _io.save(
            record_dir / "target.vtk",
            target,
            point_data={"mask": target_vert_mask.astype(np.int8)},
        )
    return source_vert_mask, target_vert_mask


def normalize(
    mesh: trimesh.Trimesh, *, centroid: npt.NDArray[np.float64], scale: float
) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_scale(1 / scale)
    mesh.apply_translation(-centroid)
    return mesh


def denormalize(
    mesh: trimesh.Trimesh, *, centroid: npt.NDArray[np.float64], scale: float
) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_translation(centroid)
    mesh.apply_scale(scale)
    return mesh
