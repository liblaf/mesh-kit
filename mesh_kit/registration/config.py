import json
import sys
from collections.abc import Sequence

import pydantic


class Params(pydantic.BaseModel):
    class Weight(pydantic.BaseModel):
        stiff: float = 0.01
        landmark: float = 10
        normal: float = 0.5

        def __add__(self, other: "Params.Weight") -> "Params.Weight":
            return Params.Weight(
                stiff=self.stiff + other.stiff,
                landmark=self.landmark + other.landmark,
                normal=self.normal + other.normal,
            )

        def __sub__(self, other: "Params.Weight") -> "Params.Weight":
            return Params.Weight(
                stiff=self.stiff - other.stiff,
                landmark=self.landmark - other.landmark,
                normal=self.normal - other.normal,
            )

        def __rmul__(self, other: int | float) -> "Params.Weight":
            return Params.Weight(
                stiff=other * self.stiff,
                landmark=other * self.landmark,
                normal=other * self.normal,
            )

        def __truediv__(self, other: int | float) -> "Params.Weight":
            return (1.0 / other) * self

    weight: Weight = pydantic.Field(default_factory=Weight)

    class Correspondence(pydantic.BaseModel):
        threshold: float = 0.1

        def __add__(self, other: "Params.Correspondence") -> "Params.Correspondence":
            return Params.Correspondence(threshold=self.threshold + other.threshold)

        def __sub__(self, other: "Params.Correspondence") -> "Params.Correspondence":
            return Params.Correspondence(threshold=self.threshold - other.threshold)

        def __rmul__(self, other: int | float) -> "Params.Correspondence":
            return Params.Correspondence(threshold=other * self.threshold)

        def __truediv__(self, other: int | float) -> "Params.Correspondence":
            return (1.0 / other) * self

    correspondence: Correspondence = pydantic.Field(default_factory=Correspondence)
    max_iter: int = 10
    eps: float = 1e-4

    def __add__(self, other: "Params") -> "Params":
        return Params(
            weight=self.weight + other.weight,
            correspondence=self.correspondence + other.correspondence,
            max_iter=self.max_iter + other.max_iter,
            eps=self.eps + other.eps,
        )

    def __sub__(self, other: "Params") -> "Params":
        return Params(
            weight=self.weight - other.weight,
            correspondence=self.correspondence - other.correspondence,
            max_iter=self.max_iter - other.max_iter,
            eps=self.eps - other.eps,
        )

    def __rmul__(self, other: int | float) -> "Params":
        return Params(
            weight=other * self.weight,
            correspondence=other * self.correspondence,
            max_iter=int(self.max_iter * other),
            eps=self.eps * other,
        )

    def __truediv__(self, other: int | float) -> "Params":
        return (1.0 / other) * self


class Config(pydantic.BaseModel):
    steps: Sequence[Params] = pydantic.Field(
        default_factory=lambda: [
            Params(weight=Params.Weight(stiff=0.01, landmark=10.0, normal=0.5)),
            Params(weight=Params.Weight(stiff=0.02, landmark=5.0, normal=0.5)),
            Params(weight=Params.Weight(stiff=0.03, landmark=2.5, normal=0.5)),
            Params(weight=Params.Weight(stiff=0.01, landmark=0.0, normal=0.0)),
        ]
    )
    gamma: float = 1.0
    watertight: bool = False


if __name__ == "__main__":
    json.dump(Config.model_json_schema(), sys.stdout)
