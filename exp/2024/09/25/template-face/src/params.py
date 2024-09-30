import dataclasses
from collections.abc import Generator, Iterable
from typing import TypedDict

import torch

import mkit.typing.numpy as nt
import mkit.typing.torch as tt


@dataclasses.dataclass(kw_only=True)
class StepParams:
    eps: float
    gamma: float
    landmark_source_idx: tt.IN
    landmark_target_pos: tt.FN3
    max_iter: int
    point_weights: tt.FN
    threshold_distance: tt.FN
    threshold_normal: tt.FN
    weight_dist: float
    weight_landmark: float
    weight_stiff: float


class StepParamsDict(TypedDict, total=False):
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
    weight_stiff: float


class PointDataDict(TypedDict):
    threshold_distance: tt.FNLike
    threshold_normal: tt.FNLike
    weight: tt.FNLike


@dataclasses.dataclass(kw_only=True)
class Params(Iterable[StepParams]):
    lr: float = 1e-4
    n_points: int
    point_data: PointDataDict = dataclasses.field(default_factory=dict)
    steps: list[StepParamsDict] = dataclasses.field(
        default_factory=lambda: [
            {"weight_landmark": 5, "weight_stiff": 50},
            {"weight_landmark": 2, "weight_stiff": 20},
            {"weight_landmark": 0.5, "weight_stiff": 5},
            {"weight_landmark": 0, "weight_stiff": 2},
            {"weight_landmark": 0, "weight_stiff": 0.8},
            {"weight_landmark": 0, "weight_stiff": 0.5},
            {"weight_landmark": 0, "weight_stiff": 0.35},
            {"weight_landmark": 0, "weight_stiff": 0.2},
        ]
    )

    def __iter__(self) -> Generator[StepParams, None, None]:
        step_dict: StepParamsDict = {}
        for step in self.steps:
            step_dict |= step
            yield self.make_step(step_dict)

    def make_step(self, step: StepParamsDict) -> StepParams:
        return StepParams(
            eps=step.get("eps", 1e-4),
            gamma=step.get("gamma", 1.0),
            landmark_source_idx=self.get_landmark_source_idx(
                step.get("landmark_source_idx")
            ),
            landmark_target_pos=self.get_landmark_target_pos(
                step.get("landmark_target_pos")
            ),
            max_iter=step.get("max_iter", 100),
            point_weights=self.get_point_weights(step.get("point_weights", 1.0)),
            threshold_distance=self.get_threshold_distance(
                step.get("threshold_distance", 0.1)
            ),
            threshold_normal=self.get_threshold_normal(
                step.get("threshold_normal", 0.5)
            ),
            weight_dist=step.get("weight_dist", 1.0),
            weight_landmark=step.get("weight_landmark", 0.0),
            weight_stiff=step.get("weight_stiff", 0.2),
        )

    def get_point_weights(self, value: tt.FLike | tt.FNLike = 1.0) -> tt.FN:
        value = torch.as_tensor(value) * torch.as_tensor(
            self.point_data.get("weight", 1.0)
        )
        return torch.broadcast_to(value, (self.n_points,))

    def get_landmark_source_idx(self, value: tt.INLike | None = None) -> tt.IN:
        if value is None:
            return torch.empty((0,), dtype=torch.int)
        return torch.as_tensor(value)

    def get_landmark_target_pos(self, value: tt.FN3Like | None = None) -> tt.FN3:
        if value is None:
            return torch.empty((0, 3))
        return torch.as_tensor(value)

    def get_threshold_distance(self, value: tt.FLike | tt.FNLike = 0.1) -> tt.FN:
        value = torch.as_tensor(value) * torch.as_tensor(
            self.point_data.get("threshold_distance", 1.0)
        )
        return torch.broadcast_to(value, (self.n_points,))

    def get_threshold_normal(self, value: tt.FLike | tt.FNLike = 0.1) -> tt.FN:
        value = torch.as_tensor(value) * torch.as_tensor(
            self.point_data.get("threshold_normal", 1.0)
        )
        return torch.broadcast_to(value, (self.n_points,))


params = Params(n_points=2)
ic(list(params))
