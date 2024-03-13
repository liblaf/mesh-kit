from typing import no_type_check

import numpy as np
import taichi as ti
from numpy import typing as npt

from mesh_kit.physics import elastic as _elastic
from mesh_kit.taichi.mesh import field as _field


@no_type_check
@ti.func
def _calc_stiffness_tet(
    V: float, Mj: ti.math.vec3, Mk: ti.math.vec3, lambda_: float, mu: float
) -> ti.math.mat3:
    return (
        1
        / (36 * V)
        * (
            lambda_ * Mk.outer_product(Mj)
            + mu * Mj.outer_product(Mk)
            + mu * Mj.dot(Mk) * ti.Matrix.identity(float, 3)
        )
    )


@ti.data_oriented
class MTM:
    mesh: ti.MeshInstance

    def __init__(self, mesh: ti.MeshInstance) -> None:
        self.mesh = mesh
        _field.place_safe(self.mesh.cells, {"lambda_": float, "mu": float})
        _field.place_safe(self.mesh.edges, {"K": ti.math.mat3})
        _field.place_safe(
            self.mesh.verts,
            {
                "f": ti.math.vec3,
                "K": ti.math.mat3,
                "pos": ti.math.vec3,
                "x": ti.math.vec3,
            },
        )

    def init(self, E: float, nu: float) -> None:
        pos: ti.MatrixField = self.mesh.verts.get_member_field("pos")
        pos.from_numpy(self.mesh.get_position_as_numpy())

        lambda_: ti.ScalarField = self.mesh.cells.get_member_field("lambda_")
        lambda_.fill(_elastic.E_nu2lambda(E, nu))
        mu: ti.ScalarField = self.mesh.cells.get_member_field("mu")
        mu.fill(_elastic.E_nu2G(E, nu))

    def calc_stiffness(self) -> None:
        K_edges: ti.MatrixField = self.mesh.edges.get_member_field("K")
        K_edges.fill(0)
        K_verts: ti.MatrixField = self.mesh.verts.get_member_field("K")
        K_verts.fill(0)
        self._calc_stiffness_kernel()

    def calc_force(self) -> None:
        f: ti.MatrixField = self.mesh.verts.get_member_field("f")
        f.fill(0)
        self._calc_force_kernel()

    def export_stiffness(self) -> npt.NDArray:
        K: npt.NDArray = np.zeros(
            shape=(len(self.mesh.verts) * 3, len(self.mesh.verts) * 3)
        )
        self._export_stiffness_kernel(K)
        return K

    @no_type_check
    @ti.kernel
    def _calc_stiffness_kernel(self):
        for c in self.mesh.cells:
            # volume
            V = (
                1
                / 6
                * ti.abs(
                    ti.Matrix.cols(
                        [c.verts[i].pos - c.verts[3].pos for i in ti.static(range(3))]
                    ).determinant()
                )
            )
            M = ti.Matrix.zero(float, 4, 3)
            for i in ti.static(range(4)):
                v = c.verts[i]
                p = [c.verts[(i + j) % 4].pos for j in range(1, 4)]
                Mi = p[0].cross(p[1]) + p[1].cross(p[2]) + p[2].cross(p[0])
                if Mi.dot(v.pos - p[0]) > 0:
                    Mi = -Mi
                M[i, :] = Mi
                v.K += _calc_stiffness_tet(V, Mi, Mi, c.lambda_, c.mu)
            for e_idx in ti.static(range(6)):
                e = c.edges[e_idx]
                v_idx = ti.Vector.zero(int, 2)
                for i in ti.static(range(2)):
                    for j in range(4):
                        if e.verts[i].id == c.verts[j].id:
                            v_idx[i] = j
                            break
                j, k = v_idx
                Mj, Mk = M[j, :], M[k, :]
                e.K += _calc_stiffness_tet(V, Mj, Mk, c.lambda_, c.mu)

    @no_type_check
    @ti.kernel
    def _calc_force_kernel(self):
        for v in self.mesh.verts:
            v.f += v.K @ v.x
            for e_idx in range(v.edges.size):
                e = v.edges[e_idx]
                if e.verts[0].id == v.id:
                    v.f += e.K @ e.verts[1].x
                elif e.verts[1].id == v.id:
                    v.f += e.K.transpose() @ e.verts[0].x

    @no_type_check
    @ti.kernel
    def _export_stiffness_kernel(self, output: ti.types.ndarray()):
        for v in self.mesh.verts:
            for i, j in ti.ndrange(3, 3):
                output[3 * v.id + i, 3 * v.id + j] = v.K[i, j]
        for e in self.mesh.edges:
            v0, v1 = e.verts[0], e.verts[1]
            for i, j in ti.ndrange(3, 3):
                output[3 * v0.id + i, 3 * v1.id + j] = e.K[i, j]
                output[3 * v1.id + i, 3 * v0.id + j] = e.K[j, i]
