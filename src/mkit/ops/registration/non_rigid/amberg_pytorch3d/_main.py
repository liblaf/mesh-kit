import functools
from typing import Any, Unpack

import pytorch3d.loss
import pytorch3d.ops
import pytorch3d.utils
import pyvista as pv
import torch
from loguru import logger
from pytorch3d.structures import Meshes

import mkit
import mkit.ops.registration.non_rigid as nr
import mkit.ops.registration.non_rigid.amberg_pytorch3d._params as p
import mkit.typing.numpy as nt
import mkit.typing.torch as tt
from mkit.ops.registration.non_rigid.amberg_pytorch3d._local_affine import LocalAffine


class Amberg(nr.NonRigidRegistrationMethod):
    live: mkit.utils.Live
    params: p.Params
    _source: pv.PolyData
    _target: pv.PolyData

    def __init__(
        self,
        source: Any,
        target: Any,
        *,
        live: mkit.utils.Live | None = None,
        **kwargs: Unpack[p.ParamsDict],
    ) -> None:
        self._source = mkit.io.pyvista.as_poly_data(source)
        self._target = mkit.io.pyvista.as_poly_data(target)
        self._target = mkit.ops.registration.preprocess.simplify_mesh(
            self._target,
            density=2
            * self._source.n_faces_strict
            * self._source.length**2
            / self._source.area,
        )
        self.params = p.Params(
            n_points=self.n_points,
            normalization_transformation=self.normalization_transformation,
            **kwargs,
        )
        if live is None:
            live = mkit.utils.Live(enable=True, save_dvc_exp=False)
        self.live = live

    def run(self) -> nr.NonRigidRegistrationResult:
        self._fitting_step()
        return nr.NonRigidRegistrationResult(points=self.result_denormalized.points)

    @functools.cached_property
    def local_affine(self) -> LocalAffine:
        return LocalAffine(self.n_points, self.source.edges_packed())

    @functools.cached_property
    def optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [{"params": self.local_affine.parameters()}],
            lr=self.params.lr,
            amsgrad=True,
        )

    @functools.cached_property
    def n_points(self) -> int:
        return self.source.num_verts_per_mesh()[0]  # pyright: ignore [reportOptionalSubscript, reportReturnType]

    @functools.cached_property
    def source(self) -> Meshes:
        source: pv.PolyData = self._source.transform(
            self.normalization_transformation, inplace=False
        )
        return mkit.io.pytorch3d.as_meshes(source)

    @functools.cached_property
    def target(self) -> Meshes:
        target: pv.PolyData = self._target.transform(
            self.normalization_transformation, inplace=False
        )
        return mkit.io.pytorch3d.as_meshes(target)

    @property
    def result(self) -> Meshes:
        points: tt.FN3 = self.local_affine(self.source.verts_packed())
        faces: tt.IN3 = self.source.faces_list()[0]
        return Meshes([points], [faces])

    @property
    def result_denormalized(self) -> pv.PolyData:
        result: pv.PolyData = mkit.io.pyvista.as_poly_data(self.result)
        result.transform(self.denorm_transformation, inplace=True)
        return result

    @functools.cached_property
    def normalization_transformation(self) -> nt.F44:
        return mkit.ops.transform.norm_transformation(self._source)

    @functools.cached_property
    def denorm_transformation(self) -> nt.F44:
        return mkit.ops.transform.denorm_transformation(self._source)

    def _find_correspondences(
        self, weight_normal: float
    ) -> tuple[tt.IN, tt.FN3, tt.FN3]:
        source_points_normals: tt.Float[torch.Tensor, "1 N 6"] = self._points_normals(
            self.result, weight_normal
        )
        target_points_normals: tt.Float[torch.Tensor, "1 N 6"] = self._points_normals(
            self.target, weight_normal
        )
        idx: tt.Integer[torch.Tensor, "1 N 1"]
        _dist, idx, _knn = pytorch3d.ops.knn_points(
            source_points_normals, target_points_normals
        )
        target_points: tt.FN3 = pytorch3d.ops.knn_gather(
            self.target.verts_padded(), idx
        ).reshape((self.n_points, 3))
        target_normals: tt.FN3 = pytorch3d.ops.knn_gather(
            self.target.verts_normals_padded(), idx
        ).reshape((self.n_points, 3))
        return idx.reshape((-1,)), target_points, target_normals

    def _points_normals(
        self, mesh: Meshes, weight_normal: float
    ) -> tt.Float[torch.Tensor, "1 N 6"]:
        return (
            torch.hstack([
                mesh.verts_packed(),
                weight_normal * mesh.verts_normals_packed(),  # pyright: ignore [reportOperatorIssue]
            ])
            .to(torch.float32)
            .reshape(1, -1, 6)
        )

    def _fitting_step(self) -> None:
        for step_params in self.params:
            self._icp_step(step_params)

    def _icp_step(self, params: p.ICPParams) -> float:
        logger.info("ICP Params: {}", params)
        last_loss: float = torch.inf
        loss: float = torch.inf
        for it in range(params.max_iter):
            _idx, target_points, target_normals = self._find_correspondences(
                params.weight_normal
            )
            loss = self._optim_step(target_points, target_normals, params.optim_params)
            if abs(loss - last_loss) / last_loss < params.eps:
                break
            last_loss = loss
        return loss

    def _optim_step(
        self,
        target_points: tt.FN3,
        target_normals: tt.FN3,
        params: p.OptimParams,
    ) -> float:
        logger.debug("Optimization Params: {}", params)
        last_loss: float = torch.inf
        loss: tt.F = torch.inf
        for it in range(params.max_iter):
            self.optimizer.zero_grad()
            points: tt.FN3 = self.local_affine(self.source.verts_packed())
            result: Meshes = self.source.update_padded(points.reshape(1, -1, 3))
            normals: tt.FN3 = result.verts_normals_packed()
            # point-to-plane distance
            distance: tt.FN = (
                ((points - target_points) * target_normals).square().sum(dim=-1)
            )
            normal_similarity: tt.FN = torch.cosine_similarity(
                normals, target_normals, dim=1
            ).abs()
            point_weights: tt.FN = params.point_weights.clone().detach()
            point_weights[distance > params.threshold_distance] = 0
            point_weights[normal_similarity < params.threshold_normal] = 0
            loss_distance: tt.F = (point_weights * distance).mean()
            # stiffness loss
            loss_stiff: tt.F = self.local_affine.loss_stiffness(gamma=params.gamma)
            # landmark loss
            loss_landmark: tt.F = torch.as_tensor(0.0)
            if len(params.landmark_source_idx) > 0:
                loss_landmark = (
                    (points[params.landmark_source_idx] - params.landmark_target_pos)
                    .square()
                    .sum(dim=1)
                    .mean()
                )
            # Laplacian smoothing
            loss_smooth: tt.F = pytorch3d.loss.mesh_laplacian_smoothing(
                Meshes([points.to(torch.float32)], self.source.faces_list()),
                method="uniform",
            )
            loss: tt.F = (
                params.weight_dist * loss_distance
                + params.weight_stiff * loss_stiff
                + params.weight_landmark * loss_landmark
                + params.weight_smooth * loss_smooth
            )
            self.live.log_metric("loss/dist", loss_distance)
            self.live.log_metric("loss/stiff", loss_stiff)
            self.live.log_metric("loss/landmark", loss_landmark)
            self.live.log_metric("loss/smooth", loss_smooth)
            self.live.log_metric("loss/sum", loss)
            if torch.abs(loss - last_loss) / last_loss < params.eps:
                self.live.next_step(milestone=True)
                break
            loss.backward()
            self.optimizer.step()
            last_loss = mkit.math.as_scalar(loss)
            self.live.next_step(milestone=(it == params.max_iter - 1))
        return mkit.math.as_scalar(loss)
