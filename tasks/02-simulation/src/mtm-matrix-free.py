import functools
import pathlib
from typing import Annotated, no_type_check

import meshio
import mkit.cli
import mkit.io
import mkit.physics.moduli
import numpy as np
import scipy.sparse.linalg
import taichi as ti
import typer
from loguru import logger
from numpy import typing as npt


@no_type_check
@ti.func
def _stiffness_tet(
    volume: float, Mj: ti.math.vec3, Mk: ti.math.vec3, lambda_: float, mu: float
) -> ti.math.mat3:
    return (
        1
        / (36 * volume)
        * (
            lambda_ * Mk.outer_product(Mj)
            + mu * Mj.outer_product(Mk)
            + mu * Mj.dot(Mk) * ti.Matrix.identity(float, 3)
        )
    )


@no_type_check
@ti.func
def is_fixed(vert: ti.template()) -> bool:
    return not is_free(vert)


@no_type_check
@ti.func
def is_free(vert: ti.template()) -> bool:
    return ti.math.isnan(vert.disp).any()


@ti.data_oriented
class MTM:
    mesh: ti.MeshInstance

    def __init__(self, mesh: meshio.Mesh) -> None:
        self.mesh = mkit.io.as_taichi(mesh, ["CE", "CV", "EV", "VE"])
        self.mesh.cells.place({"lambda_": float, "mu": float, "volume": float})
        self.mesh.edges.place({"K": ti.math.mat3})
        self.mesh.verts.place(
            {
                "Ax": ti.math.vec3,
                "b": ti.math.vec3,
                "disp": ti.math.vec3,
                "f": ti.math.vec3,
                "K": ti.math.mat3,
                "pos": ti.math.vec3,
                "x": ti.math.vec3,
            }
        )
        disp: ti.MatrixField = self.mesh.verts.get_member_field("disp")
        disp.from_numpy(mesh.point_data["disp"])
        pos: ti.MatrixField = self.mesh.verts.get_member_field("pos")
        pos.from_numpy(self.mesh.get_position_as_numpy())

    @functools.cached_property
    def fixed_mask(self) -> npt.NDArray[np.bool_]:
        return ~self.free_mask

    @functools.cached_property
    def free_mask(self) -> npt.NDArray[np.bool_]:
        disp_ti: ti.MatrixField = self.mesh.verts.get_member_field("disp")
        disp_np: npt.NDArray[np.floating] = disp_ti.to_numpy()
        fixed_mask: npt.NDArray[np.bool_] = np.isnan(disp_np).any(axis=1)
        return fixed_mask

    def init_material(self, E: float, nu: float) -> None:
        lambda_: float = mkit.physics.moduli.E_nu2lambda(E, nu)
        lambda_ti: ti.ScalarField = self.mesh.cells.get_member_field("lambda_")
        lambda_ti.fill(lambda_)
        mu: float = mkit.physics.moduli.E_nu2mu(E, nu)
        mu_ti: ti.ScalarField = self.mesh.cells.get_member_field("mu")
        mu_ti.fill(mu)

    def volume(self) -> None:
        self._volume()

    def stiffness(self) -> None:
        K_edges: ti.MatrixField = self.mesh.edges.get_member_field("K")
        K_edges.fill(0)
        K_verts: ti.MatrixField = self.mesh.verts.get_member_field("K")
        K_verts.fill(0)
        self._stiffness()

    def force(self, x_free: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        disp_ti: ti.MatrixField = self.mesh.verts.get_member_field("disp")
        x_np: npt.NDArray[np.floating] = disp_ti.to_numpy()
        x_np[self.free_mask] = x_free
        x_ti: ti.MatrixField = self.mesh.verts.get_member_field("x")
        x_ti.from_numpy(x_np)
        self._force()
        f_ti: ti.MatrixField = self.mesh.verts.get_member_field("f")
        f_np: npt.NDArray[np.floating] = f_ti.to_numpy()
        return f_np

    def Ax(self, x_free: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        disp_ti: ti.MatrixField = self.mesh.verts.get_member_field("disp")
        x_np: npt.NDArray[np.floating] = disp_ti.to_numpy()
        x_np[self.free_mask] = x_free
        x_ti: ti.MatrixField = self.mesh.verts.get_member_field("x")
        x_ti.from_numpy(x_np)
        self._Ax()
        Ax_ti: ti.MatrixField = self.mesh.verts.get_member_field("Ax")
        Ax_np: npt.NDArray[np.floating] = Ax_ti.to_numpy()
        return Ax_np[self.free_mask]

    def b(self) -> npt.NDArray[np.floating]:
        disp_ti: ti.MatrixField = self.mesh.verts.get_member_field("disp")
        x_ti: ti.MatrixField = self.mesh.verts.get_member_field("x")
        x_ti.copy_from(disp_ti)
        self._b()
        b_ti: ti.MatrixField = self.mesh.verts.get_member_field("b")
        b_np: npt.NDArray[np.floating] = b_ti.to_numpy()
        return b_np[self.free_mask]

    @no_type_check
    @ti.kernel
    def _volume(self):
        for c in self.mesh.cells:
            dX = ti.Matrix.cols(
                [c.verts[i].pos - c.verts[3].pos for i in ti.static(range(3))]
            )
            c.volume = 1 / 6 * ti.abs(dX.determinant())

    @no_type_check
    @ti.kernel
    def _stiffness(self):
        for c in self.mesh.cells:
            M = ti.Matrix.zero(ti.f64, 4, 3)
            for i in ti.static(range(4)):
                v = c.verts[i]
                p = [c.verts[(i + j) % 4].pos for j in range(1, 4)]
                Mi = p[0].cross(p[1]) + p[1].cross(p[2]) + p[2].cross(p[0])
                if Mi.dot(v.pos - p[0]) > 0:
                    Mi = -Mi
                M[i, :] = Mi
                v.K += _stiffness_tet(c.volume, Mi, Mi, c.lambda_, c.mu)
            for e_id in ti.static(range(6)):
                e = c.edges[e_id]
                v_idx = ti.Vector.zero(int, 2)  # vertex index in cell (in range(4))
                for i in ti.static(range(2)):
                    for j in range(4):
                        if e.verts[i].id == c.verts[j].id:
                            v_idx[i] = j
                            break
                j, k = v_idx
                Mj, Mk = M[j, :], M[k, :]
                e.K += _stiffness_tet(c.volume, Mj, Mk, c.lambda_, c.mu)

    @no_type_check
    @ti.kernel
    def _force(self):
        for v in self.mesh.verts:
            v.f = v.K @ v.x
            for e_idx in range(v.edges.size):
                e = v.edges[e_idx]
                if e.verts[0].id == v.id:
                    v.f += e.K @ e.verts[1].x
                elif e.verts[1].id == v.id:
                    v.f += e.K.transpose() @ e.verts[0].x

    @no_type_check
    @ti.kernel
    def _Ax(self):
        for v in self.mesh.verts:
            if is_free(v):
                v.Ax = v.K @ v.x
                for e_idx in range(v.edges.size):
                    e = v.edges[e_idx]
                    if e.verts[0].id == v.id and is_free(e.verts[1]):
                        v.Ax += e.K @ e.verts[1].x
                    elif e.verts[1].id == v.id and is_free(e.verts[0]):
                        v.Ax += e.K.transpose() @ e.verts[0].x

    @no_type_check
    @ti.kernel
    def _b(self):
        for v in self.mesh.verts:
            if is_free(v):
                v.b = 0
                for e_idx in range(v.edges.size):
                    e = v.edges[e_idx]
                    if e.verts[0].id == v.id and is_fixed(e.verts[1]):
                        v.b -= e.K @ e.verts[1].disp
                    elif e.verts[1].id == v.id and is_fixed(e.verts[0]):
                        v.b -= e.K.transpose() @ e.verts[0].disp


class Operator(scipy.sparse.linalg.LinearOperator):
    mtm: MTM

    def __init__(self, mtm: MTM) -> None:
        self.mtm = mtm
        super().__init__(float, (self.n_free * 3, self.n_free * 3))

    @functools.cached_property
    def n_free(self) -> int:
        return np.count_nonzero(self.mtm.free_mask)

    def _matvec(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return self.mtm.Ax(x.reshape((self.n_free, 3))).flatten()


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
    poisson_ratio: Annotated[float, typer.Option()] = 0.0,
) -> None:
    ti.init(ti.cpu, default_fp=ti.float64)
    mesh: meshio.Mesh = mkit.io.load_meshio(input_file)
    mtm = MTM(mesh)
    mtm.init_material(E=3000, nu=poisson_ratio)
    mtm.volume()
    mtm.stiffness()
    A = Operator(mtm)
    x_free: npt.NDArray[np.floating]
    info: int
    x_free, info = scipy.sparse.linalg.gmres(A, mtm.b().flatten())
    x_free = x_free.reshape((A.n_free, 3))
    logger.info("GMRES info: {}", info)
    disp: npt.NDArray[np.floating] = mesh.point_data["disp"]
    mesh.points[mtm.fixed_mask] += disp[mtm.fixed_mask]
    mesh.points[mtm.free_mask] += x_free
    mkit.io.save(output_file, mesh, point_data={"force": mtm.force(x_free)})


if __name__ == "__main__":
    mkit.cli.run(main)
