import math
from typing import no_type_check

import meshio
import numpy as np
import taichi as ti
import torch
import torch.autograd.gradcheck
from icecream import ic
from mkit import io as _io
from numpy import typing as npt

ti.init(arch=ti.cuda, default_fp=ti.f64, debug=True, verbose=True)
torch.set_default_device("cuda")
torch.set_default_dtype(torch.float64)


@ti.data_oriented
class Hyperelastic:
    mesh: ti.MeshInstance

    def __init__(self, mesh: ti.MeshInstance) -> None:
        self.mesh = mesh
        self.mesh.cells.place({"F": ti.math.mat3, "F_grad": ti.math.mat3, "V": ti.f64})
        self.mesh.verts.place({"pos": ti.math.vec3})
        self.mesh.verts.place({"x": ti.math.vec3, "x_grad": ti.math.vec3})
        pos: ti.MatrixField = self.mesh.verts.get_member_field("pos")
        pos.from_numpy(mesh.get_position_as_numpy())

    def deformation_gradient(self) -> None:
        self._deformation_gradient()

    def deformation_gradient_grad(self) -> None:
        self._deformation_gradient_grad()

    @no_type_check
    @ti.kernel
    def _deformation_gradient(self):
        """https://en.wikipedia.org/wiki/Finite_strain_theory#Deformation_gradient_tensor"""
        for c in self.mesh.cells:
            dX = ti.Matrix.cols(
                [c.verts[i].pos - c.verts[3].pos for i in ti.static(range(3))]
            )
            pos_new = [c.verts[i].pos + c.verts[i].x for i in ti.static(range(4))]
            dx = ti.Matrix.cols([pos_new[i] - pos_new[3] for i in ti.static(range(3))])
            c.F = dx @ dX.inverse()

    @no_type_check
    @ti.kernel
    def _deformation_gradient_grad(self):
        for c in self.mesh.cells:
            dX = ti.Matrix.cols(
                [c.verts[i].pos - c.verts[3].pos for i in ti.static(range(3))]
            )
            grad = c.F_grad @ dX.inverse().transpose()
            c.verts[0].x_grad += grad[:, 0]
            c.verts[1].x_grad += grad[:, 1]
            c.verts[2].x_grad += grad[:, 2]
            c.verts[3].x_grad -= grad[:, 0] + grad[:, 1] + grad[:, 2]

    @no_type_check
    @ti.kernel
    def _volume(self):
        for c in self.mesh.cells:
            dX = ti.Matrix.cols(
                [c.verts[i].pos - c.verts[3].pos for i in ti.static(range(3))]
            )
            c.V = ti.abs(dX.determinant()) / 6


class DeformationGradient(torch.autograd.Function):
    @staticmethod
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        model: Hyperelastic, x: torch.Tensor
    ) -> torch.Tensor:
        x_ti: ti.MatrixField = model.mesh.verts.get_member_field("x")
        x_ti.from_torch(x)
        model.deformation_gradient()
        F_ti: ti.MatrixField = model.mesh.cells.get_member_field("F")
        return F_ti.to_torch()

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.BackwardCFunction,
        inputs: tuple[Hyperelastic, torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        model: Hyperelastic
        x: torch.Tensor
        model, x = inputs
        ctx.metadata["model"] = model
        ctx.save_for_backward(x)

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: torch.autograd.function.BackwardCFunction, F_grad: torch.Tensor
    ) -> tuple[None, torch.Tensor]:
        model: Hyperelastic = ctx.metadata["model"]
        x: torch.Tensor = ctx.saved_tensors[0]
        x_ti: ti.MatrixField = model.mesh.verts.get_member_field("x")
        x_ti.from_torch(x)
        F_grad_ti: ti.MatrixField = model.mesh.cells.get_member_field("F_grad")
        F_grad_ti.from_torch(F_grad)
        x_grad_ti: ti.MatrixField = model.mesh.verts.get_member_field("x_grad")
        x_grad_ti.fill(0)
        model.deformation_gradient_grad()
        return None, x_grad_ti.to_torch("cuda")


mesh_io: meshio.Mesh = meshio.read("sphere.vtu")
mesh_ti: ti.MeshInstance = _io.to_taichi(mesh_io, relations=["CV"])
model = Hyperelastic(mesh_ti)


def deformation_gradient(x: torch.Tensor) -> torch.Tensor:
    return DeformationGradient.apply(model, x)


disp_np: npt.NDArray = mesh_io.point_data["disp"]
free_mask: npt.NDArray = np.any(np.isnan(disp_np), axis=1)
fixed_mask: npt.NDArray = ~free_mask
num_free: int = np.count_nonzero(free_mask)
num_fixed: int = np.count_nonzero(fixed_mask)
num_verts: int = len(mesh_ti.verts)
assert num_free + num_fixed == num_verts
x_free = torch.zeros(num_free, 3, requires_grad=True)
# disp_np[fixed_mask] = 0
x_fixed = torch.tensor(disp_np[fixed_mask])
model._volume()
V_ti: ti.ScalarField = mesh_ti.cells.get_member_field("V")
V: torch.Tensor = V_ti.to_torch("cuda")
num_cells: int = len(mesh_ti.cells)
volume: ti.MatrixField = mesh_ti.verts.get_member_field("V")

E = 3000
nu = 0.46
lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
optimizer = torch.optim.RMSprop([x_free], lr=0.1)
origin_loss: float = torch.nan
last_loss: float = torch.nan
for i in range(5000):
    optimizer.zero_grad()
    x = torch.zeros(len(mesh_io.points), 3)
    x[free_mask] = x_free
    x[fixed_mask] = x_fixed
    F = deformation_gradient(x)
    # ic(F.shape)

    C1 = mu / 2
    D1 = lambda_ / 2
    C = torch.bmm(F.transpose(1, 2), F)
    # ic(C.shape)
    I1 = torch.vmap(torch.trace)(C)
    # ic(I1.shape)
    J = torch.vmap(torch.linalg.det)(F)
    ic(torch.isnan(torch.log(J)).any())
    # ic(J.shape)
    # W = C1 * (I1 - 3 - 2 * torch.log(J)) + D1 * (J - 1) ** 2
    W = C1 * (I1 - 3)
    # W = torch.abs(W)

    # I = torch.eye(3).view(1, 3, 3).repeat(num_cells, 1, 1)
    # E = 0.5 * (torch.bmm(F.transpose(1, 2), F) - I)
    # W = lambda_ / 2 * torch.vmap(torch.trace)(E) ** 2 + mu * torch.vmap(torch.trace)(
    #     torch.bmm(E, E)
    # )

    energy = torch.sum(V * W)
    loss: float = energy.item()
    ic(i, loss)
    if math.isnan(origin_loss):
        origin_loss = loss
    # if abs(last_loss - loss) / origin_loss < 1e-8:
    #     break
    last_loss = loss
    energy.backward()
    # optimizer.step()
    with torch.no_grad():
        x_free -= 1e-5 * x_free.grad
mesh_io.points[free_mask] += x_free.detach().cpu().numpy()
mesh_io.points[fixed_mask] += disp_np[fixed_mask]
mesh_io.write("hookean-0.50.vtu")
