import numpy as np
import torch

import mkit.typing.numpy as nt
import mkit.typing.torch as tt


class LocalAffine(torch.nn.Module):
    A: tt.Float[torch.Tensor, "V 3 3"]
    b: tt.FN3
    edges: nt.IN2
    n_points: int

    def __init__(self, n_points: int, edges: nt.IN2Like) -> None:
        super().__init__()
        self.A = torch.nn.Parameter(torch.eye(3, 3).tile(n_points, 1, 1))
        self.b = torch.nn.Parameter(torch.zeros(n_points, 3))
        self.edges = np.asarray(edges)
        self.n_points = n_points

    def forward(self, points: tt.FN3Like) -> tt.FN3:
        points: tt.FN3 = torch.as_tensor(points).reshape((self.n_points, 3))
        points = torch.bmm(
            self.A.to(points.dtype), points.reshape((self.n_points, 3, 1))
        ).reshape((
            self.n_points,
            3,
        ))
        points += self.b
        return points

    @property
    def stiffness(self) -> tt.Float[torch.Tensor, "E 3 4"]:
        transform: tt.Float[torch.Tensor, "V 3 4"] = torch.cat(
            [self.A, self.b.reshape((self.n_points, 3, 1))], dim=2
        )
        stiffness: tt.Float[torch.Tensor, "E 3 4"] = (
            transform[self.edges[:, 0]] - transform[self.edges[:, 1]]
        )
        return stiffness

    def loss_stiffness(self, gamma: float = 1.0) -> tt.F:
        stiff: tt.Float[torch.Tensor, "E 3 4"] = self.stiffness
        stiff *= torch.as_tensor([1, 1, 1, gamma])
        return stiff.square().sum(dim=(1, 2)).mean()
