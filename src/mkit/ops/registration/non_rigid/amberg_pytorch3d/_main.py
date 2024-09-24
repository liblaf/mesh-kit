from typing import Any

import pytorch3d
import pytorch3d.ops
import torch
from dvclive import Live
from pytorch3d.structures import Meshes

import mkit
import mkit.typing.torch as tt
from mkit.ops.registration.non_rigid.amberg_pytorch3d._local_affine import LocalAffine
from mkit.ops.registration.non_rigid.amberg_pytorch3d._params import (
    ParamsDict,
    ParamsSchema,
    StepParams,
)
from mkit.ops.registration.non_rigid.amberg_pytorch3d._result import (
    NonRigidRegistrationResult,
)


def nricp_amberg_pytorch3d(
    source: Any, target: Any, params: ParamsDict | None = None
) -> NonRigidRegistrationResult:
    source: Meshes = mkit.io.pytorch3d.as_meshes(source)
    target: Meshes = mkit.io.pytorch3d.as_meshes(target)
    n_points: int = source.num_verts_per_mesh()[0]  # pyright: ignore [reportOptionalSubscript]
    params: ParamsSchema = ParamsSchema(n_points=n_points, **(params or {}))
    with Live(monitor_system=True) as live:
        local_affine: LocalAffine = LocalAffine(n_points, source.edges_packed())  # pyright: ignore [reportArgumentType]
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            [{"params": local_affine.parameters()}], lr=params.lr, amsgrad=True
        )
        _loop(
            source=source,
            target=target,
            local_affine=local_affine,
            optimizer=optimizer,
            params=params,
            live=live,
        )
        live.make_summary()
    points: tt.FN3 = local_affine(source.verts_packed())
    return NonRigidRegistrationResult(result=points.numpy(force=True))


def _loop(
    source: Meshes,
    target: Meshes,
    local_affine: LocalAffine,
    optimizer: torch.optim.Optimizer,
    params: ParamsSchema,
    live: Live,
) -> None:
    n_points: int = source.num_verts_per_mesh()[0]  # pyright: ignore [reportOptionalSubscript]
    for i in range(len(params.steps)):
        points: tt.FN3 = local_affine(source.verts_packed())
        _dists, idx, _knn = pytorch3d.ops.knn_points(
            points.reshape((1, n_points, 3)).to(torch.float32),
            target.verts_padded().to(torch.float32),  # pyright: ignore [reportOptionalMemberAccess, reportAttributeAccessIssue]
        )
        target_points: tt.FN3 = pytorch3d.ops.knn_gather(
            target.verts_padded(),  # pyright: ignore [reportArgumentType]
            idx,
        ).reshape((n_points, 3))
        target_normals: tt.FN3 = pytorch3d.ops.knn_gather(
            target.verts_normals_padded(), idx
        ).reshape((n_points, 3))
        _step(
            source=source,
            target_points=target_points,
            target_normals=target_normals,
            local_affine=local_affine,
            optimizer=optimizer,
            params=params.step(i).make(),
            live=live,
        )


def _step(
    source: Meshes,
    target_points: tt.FN3,
    target_normals: tt.FN3,
    local_affine: LocalAffine,
    optimizer: torch.optim.Optimizer,
    params: StepParams,
    live: Live,
) -> None:
    last_loss: tt.F = torch.inf
    for _ in range(params.max_iter):
        ic(_)
        optimizer.zero_grad()
        points: tt.FN3 = local_affine(source.verts_packed())
        result: Meshes = source.update_padded(points.reshape(1, -1, 3))
        normals: tt.FN3 = result.verts_normals_packed()
        # point-to-plane distance
        distance: tt.FN = (
            ((points - target_points) * target_normals).square().sum(dim=-1)
        )
        normal_similarity: tt.FN = torch.cosine_similarity(
            normals, target_normals, dim=1
        ).abs()
        point_weights: tt.FN = torch.tensor(params.point_weights)
        point_weights[distance > params.threshold_distance] = 0
        point_weights[normal_similarity < params.threshold_normal] = 0
        loss_distance: tt.F = (point_weights * distance).mean()
        # stiffness loss
        loss_stiff: tt.F = local_affine.loss_stiffness(gamma=params.gamma)
        # landmark loss
        loss_landmark: tt.F = torch.as_tensor(0.0)
        if params.source_landmark_idx:
            loss_landmark = (
                (points[params.source_landmark_idx] - params.target_landmark_pos)
                .square()
                .sum(dim=1)
                .mean()
            )
        loss: tt.F = (
            params.weight_dist * loss_distance
            + params.weight_stiff * loss_stiff
            + params.weight_landmark * loss_landmark
        )
        live.log_metric("loss/dist", loss_distance.item())
        live.log_metric("loss/stiff", loss_stiff.item())
        live.log_metric("loss/landmark", loss_landmark.item())
        live.log_metric("loss/sum", loss.item())
        if torch.abs(loss - last_loss) / last_loss < params.eps:
            break
        loss.backward()
        optimizer.step()
        live.next_step()
