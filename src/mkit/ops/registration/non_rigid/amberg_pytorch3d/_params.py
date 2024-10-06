import dataclasses
from collections.abc import Generator, Iterable
from typing import TypedDict

import torch
import trimesh.transformations as tr

import mkit
import mkit.typing.numpy as nt
import mkit.typing.torch as tt


@dataclasses.dataclass(kw_only=True)
class OptimParams:
    eps: float = 1e-4
    gamma: float = 1.0
    landmark_source_idx: tt.IN
    landmark_target_pos: tt.FN3
    max_iter: int = 100
    point_weights: tt.FN
    threshold_distance: tt.FN
    threshold_normal: tt.FN
    weight_dist: float = 1.0
    weight_landmark: float = 0.0
    weight_smooth: float = 0.0
    weight_stiff: float = 0.1


class OptimParamsDict(TypedDict, total=False):
    eps: float
    gamma: float
    landmark_source_idx: nt.INLike
    landmark_target_pos: nt.FN3Like
    max_iter: int
    point_weights: nt.FNLike
    threshold_distance: tt.FNLike
    threshold_normal: tt.FNLike
    weight_dist: float
    weight_landmark: float
    weight_smooth: float
    weight_stiff: float


@dataclasses.dataclass(kw_only=True)
class ICPParams:
    eps: float = 1e-4
    max_iter: int = 7
    optim_params: OptimParams
    weight_normal: float = 0.0


class ICPParamsDict(TypedDict, total=False):
    eps: float
    max_iter: int
    optim_params: OptimParamsDict
    weight_normal: float


class PointDataDict(TypedDict, total=False):
    threshold_distance: tt.FNLike
    threshold_normal: tt.FNLike
    weight: tt.FNLike


class ParamsDict(TypedDict, total=False):
    lr: float
    point_data: PointDataDict
    steps: Iterable[ICPParamsDict]


@dataclasses.dataclass(kw_only=True)
class Params(Iterable[ICPParams]):
    lr: float = 1e-4
    n_points: int
    normalization_transformation: nt.F44
    point_data: PointDataDict = dataclasses.field(default_factory=dict)
    steps: Iterable[ICPParamsDict] = dataclasses.field(
        default_factory=lambda: [
            {"optim_params": {"weight_landmark": 5, "weight_stiff": 50}},
            {"optim_params": {"weight_landmark": 2, "weight_stiff": 20}},
            {"optim_params": {"weight_landmark": 0.5, "weight_stiff": 5}},
            {"optim_params": {"weight_landmark": 0, "weight_stiff": 2}},
            {"optim_params": {"weight_landmark": 0, "weight_stiff": 0.8}},
            {"optim_params": {"weight_landmark": 0, "weight_stiff": 0.5}},
            {"optim_params": {"weight_landmark": 0, "weight_stiff": 0.35}},
            {"optim_params": {"weight_landmark": 0, "weight_stiff": 0.2}},
            {"optim_params": {"weight_landmark": 0, "weight_stiff": 0.1}},
        ]
    )

    def __iter__(self) -> Generator[ICPParams, None, None]:
        icp_params_dict: ICPParamsDict = {}
        for step in self.steps:
            icp_params_dict = mkit.utils.merge_mapping(icp_params_dict, step)
            yield self.make_step_params(icp_params_dict)

    def make_step_params(self, kwargs: ICPParamsDict) -> ICPParams:
        return ICPParams(
            eps=kwargs.get("eps", 1e-4),
            max_iter=kwargs.get("max_iter", 7),
            optim_params=self.make_optim_params(kwargs.get("optim_params", {})),
            weight_normal=kwargs.get("weight_normal", 0.0),
        )

    def make_optim_params(self, kwargs: OptimParamsDict) -> OptimParams:
        return OptimParams(
            eps=kwargs.get("eps", 1e-4),
            gamma=kwargs.get("gamma", 1.0),
            landmark_source_idx=self.get_landmark_source_idx(
                kwargs.get("landmark_source_idx")
            ),
            landmark_target_pos=self.get_landmark_target_pos(
                kwargs.get("landmark_target_pos")
            ),
            max_iter=kwargs.get("max_iter", 100),
            point_weights=self.get_point_weights(kwargs.get("point_weights", 1.0)),
            threshold_distance=self.get_threshold_distance(
                kwargs.get("threshold_distance", 0.1)
            ),
            threshold_normal=self.get_threshold_normal(
                kwargs.get("threshold_normal", 0.5)
            ),
            weight_dist=kwargs.get("weight_dist", 1.0),
            weight_landmark=kwargs.get("weight_landmark", 0.0),
            weight_smooth=kwargs.get("weight_smooth", 0.0),
            weight_stiff=kwargs.get("weight_stiff", 0.1),
        )

    def get_landmark_source_idx(self, value: tt.INLike | None = None) -> tt.IN:
        if value is None:
            return torch.empty((0,), dtype=torch.int)
        return torch.as_tensor(value)

    def get_landmark_target_pos(self, value: tt.FN3Like | None = None) -> tt.FN3:
        if value is None:
            return torch.empty((0, 3))
        return torch.as_tensor(
            tr.transform_points(value, self.normalization_transformation)
        )

    def get_threshold_distance(self, value: tt.FLike | tt.FNLike = 0.1) -> tt.FN:
        value = torch.as_tensor(value) * torch.as_tensor(
            self.point_data.get("threshold_distance", 1.0)
        )
        return torch.broadcast_to(value, (self.n_points,))

    def get_threshold_normal(self, value: tt.FLike | tt.FNLike = 0.5) -> tt.FN:
        value = torch.as_tensor(value) * torch.as_tensor(
            self.point_data.get("threshold_normal", 1.0)
        )
        return torch.broadcast_to(value, (self.n_points,))

    def get_point_weights(self, value: tt.FLike | tt.FNLike = 1.0) -> tt.FN:
        value = torch.as_tensor(value) * torch.as_tensor(
            self.point_data.get("weight", 1.0)
        )
        return torch.broadcast_to(value, (self.n_points,))
