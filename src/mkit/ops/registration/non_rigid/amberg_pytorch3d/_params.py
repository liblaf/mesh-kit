import dataclasses
from collections.abc import Iterable, Sequence
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import torch

import mkit
import mkit.typing.numpy as nt
import mkit.typing.torch as tt

mkit.logging.init()


@dataclasses.dataclass(kw_only=True)
class StepParams:
    eps: float
    gamma: float
    max_iter: int
    point_weights: nt.FN
    source_landmark_idx: nt.IN
    target_landmark_pos: nt.FN3
    threshold_distance: tt.FN
    threshold_normal: tt.FN
    weight_dist: float
    weight_landmark: float
    weight_stiff: float


class StepParamsDict(TypedDict, total=False):
    eps: float
    gamma: float
    max_iter: int
    point_weights: nt.FNLike
    source_landmark_idx: nt.INLike
    target_landmark_pos: nt.FN3Like
    threshold_distance: tt.FNLike
    threshold_normal: tt.FNLike
    weight_dist: float
    weight_landmark: float
    weight_stiff: float


@dataclasses.dataclass(kw_only=True)
class StepParamsSchema:
    eps: float = 1e-4
    gamma: float = 1
    max_iter: int = 100
    n_points: int
    point_weights: nt.FNLike = 1.0
    source_landmark_idx: nt.INLike | None = None
    target_landmark_pos: nt.FN3Like | None = None
    threshold_distance: tt.FNLike | None = 0.1
    threshold_normal: tt.FNLike = 0.5
    weight_dist: float = 1.0
    weight_landmark: float = 0.0
    weight_stiff: float = 0.2

    def make(self) -> StepParams:
        return StepParams(
            eps=self.eps,
            gamma=self.gamma,
            max_iter=self.max_iter,
            point_weights=self.get_point_weights(),
            source_landmark_idx=self.get_source_landmark_idx(),
            target_landmark_pos=self.get_target_landmark_pos(),
            threshold_distance=self.get_threshold_distance(),
            threshold_normal=self.get_threshold_normal(),
            weight_dist=self.weight_dist,
            weight_landmark=self.weight_landmark,
            weight_stiff=self.weight_stiff,
        )

    def get_point_weights(self) -> nt.FN:
        return self._broadcast(
            self.point_weights, default=1.0, shape=(self.n_points,), dtype=float
        )

    def get_source_landmark_idx(self) -> nt.IN:
        if self.source_landmark_idx is None:
            return np.empty((0,), dtype=int)
        return np.asarray(self.source_landmark_idx, int).flatten()

    def get_target_landmark_pos(self) -> nt.FN3:
        if self.target_landmark_pos is None:
            return np.empty((0, 3), dtype=float)
        return np.asarray(self.target_landmark_pos, float).reshape((-1, 3))

    def get_threshold_distance(self) -> tt.FN:
        return torch.tensor(
            self._broadcast(
                self.threshold_distance,
                default=0.1,
                shape=(self.n_points,),
                dtype=float,
            )
        )

    def get_threshold_normal(self) -> tt.FN:
        return torch.tensor(
            self._broadcast(
                self.threshold_normal, default=0.5, shape=(self.n_points,), dtype=float
            )
        )

    def _broadcast(
        self,
        value: npt.ArrayLike | None,
        *,
        default: float | None = None,
        dtype: npt.DTypeLike | None = None,
        shape: int | Sequence[int] | None = None,
    ) -> npt.NDArray[...]:
        if value is None:
            value = default
        value = np.asarray(value)
        if dtype is not None:
            value = value.astype(dtype)
        if shape is not None:
            value = np.broadcast_to(value.squeeze(), shape)
        return value


class ParamsDict(TypedDict, total=False):
    default: StepParamsDict
    lr: float
    steps: list[StepParamsDict]


@dataclasses.dataclass(kw_only=True)
class ParamsSchema:
    default: StepParamsDict
    lr: float = 1e-4
    n_points: int
    steps: list[StepParamsDict]

    def __init__(
        self,
        *,
        default: StepParamsDict | None = None,
        lr: float = 1e-4,
        n_points: int,
        steps: Iterable[StepParamsDict] | None = None,
    ) -> None:
        self.default = default or {}
        self.lr = lr
        self.n_points = n_points
        self.steps = list(
            steps
            or [
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

    def step(self, idx: int) -> StepParamsSchema:
        return StepParamsSchema(
            n_points=self.n_points, **(self.default | self.steps[idx])
        )
