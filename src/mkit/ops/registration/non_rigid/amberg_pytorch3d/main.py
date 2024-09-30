from typing import Any, Unpack

import pytorch3d.ops
import torch
from pytorch3d.structures import Meshes

import mkit
import mkit.ops.registration.non_rigid as nr
import mkit.ops.registration.non_rigid.amberg_pytorch3d._params as p
import mkit.typing.torch as tt
from mkit.ops.registration.non_rigid.amberg_pytorch3d._local_affine import LocalAffine


class Amberg(nr.NonRigidRegistrationMethod):
    live: mkit.utils.Live
    params: p.Params
    source: Meshes
    target: Meshes

    def __init__(
        self,
        source: Any,
        target: Any,
        *,
        live: mkit.utils.Live | None = None,
        **kwargs: Unpack[p.ParamsDict],
    ) -> None:
        self.source = mkit.io.pytorch3d.as_meshes(source)
        self.target = mkit.io.pytorch3d.as_meshes(target)
        self.params = p.Params(n_points=self.n_points, **kwargs)  # pyright: ignore [reportArgumentType, reportOptionalSubscript]
        if live is None:
            live = mkit.utils.Live(enable=False)
        self.live = live

    def run(self) -> nr.NonRigidRegistrationResult:
        raise NotImplementedError

    @property
    def n_points(self) -> int:
        return self.source.num_verts_per_mesh()[0]  # pyright: ignore [reportOptionalSubscript, reportReturnType]

    def _loop(
        self,
        source: Meshes,
        target: Meshes,
        local_affine: LocalAffine,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        for step_params in self.params:
            points: tt.FN3 = local_affine(source.verts_packed())
            _dists, idx, _knn = pytorch3d.ops.knn_points(
                points.reshape((1, self.n_points, 3)).to(torch.float32),
                target.verts_padded().to(torch.float32),  # pyright: ignore [reportOptionalMemberAccess, reportAttributeAccessIssue]
            )
            target_points: tt.FN3 = pytorch3d.ops.knn_gather(
                target.verts_padded(),  # pyright: ignore [reportArgumentType]
                idx,
            ).reshape((self.n_points, 3))
            target_normals: tt.FN3 = pytorch3d.ops.knn_gather(
                target.verts_normals_padded(), idx
            ).reshape((self.n_points, 3))
            self._step(
                source=source,
                target_points=target_points,
                target_normals=target_normals,
                local_affine=local_affine,
                optimizer=optimizer,
                params=step_params,
            )

    def _step(
        self,
        source: Meshes,
        target_points: tt.FN3,
        target_normals: tt.FN3,
        local_affine: LocalAffine,
        optimizer: torch.optim.Optimizer,
        params: p.StepParams,
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
            if params.landmark_source_idx:
                loss_landmark = (
                    (points[params.landmark_source_idx] - params.landmark_target_pos)
                    .square()
                    .sum(dim=1)
                    .mean()
                )
            loss: tt.F = (
                params.weight_dist * loss_distance
                + params.weight_stiff * loss_stiff
                + params.weight_landmark * loss_landmark
            )
            self.live.log_metric("loss/dist", loss_distance.item())
            self.live.log_metric("loss/stiff", loss_stiff.item())
            self.live.log_metric("loss/landmark", loss_landmark.item())
            self.live.log_metric("loss/sum", loss.item())
            if torch.abs(loss - last_loss) / last_loss < params.eps:
                break
            loss.backward()
            optimizer.step()
            self.live.next_step()
