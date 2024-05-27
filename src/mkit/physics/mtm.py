import pathlib
import tempfile
from collections.abc import Iterable
from typing import no_type_check

import numpy as np
import scipy
import scipy.io
import scipy.sparse
import taichi as ti
from numpy import typing as npt

import mkit
import mkit.io
import mkit.physics.moduli
import mkit.taichi.mesh.field
from mkit.io.types import AnyMesh


@ti.data_oriented
class MTM:
    mesh: ti.MeshInstance

    def __init__(self, mesh: AnyMesh, relations: Iterable[str] = []) -> None:
        self.mesh = mkit.io.as_taichi(mesh, relations)
        mkit.taichi.mesh.field.place(self.mesh.verts, {"pos": ti.math.vec3})
        pos: ti.MatrixField = self.mesh.verts.get_member_field("pos")
        pos.from_numpy(self.mesh.get_position_as_numpy())

    @property
    def n_verts(self) -> int:
        return len(self.mesh.verts)

    def init_material(self, E: float, nu: float) -> None:
        """
        Args:
            E: Young's modulus
            nu: Poisson's ratio
        """
        mkit.taichi.mesh.field.place(self.mesh.cells, {"lambda_": float, "mu": float})
        lambda_: float = mkit.physics.moduli.E_nu2lambda(E, nu)
        lambda_ti: ti.ScalarField = self.mesh.cells.get_member_field("lambda_")
        lambda_ti.fill(lambda_)
        mu: float = mkit.physics.moduli.E_nu2mu(E, nu)
        mu_ti: ti.ScalarField = self.mesh.cells.get_member_field("mu")
        mu_ti.fill(mu)

    def volume(self) -> npt.NDArray[np.floating]:
        mkit.taichi.mesh.field.place(self.mesh.cells, {"vol": float})
        self._volume()
        volume_ti: ti.ScalarField = self.mesh.cells.get_member_field("vol")
        volume_np: npt.NDArray[np.floating] = volume_ti.to_numpy()
        return volume_np

    def stiffness(self) -> None:
        mkit.taichi.mesh.field.place(self.mesh.edges, {"K": ti.math.mat3})
        mkit.taichi.mesh.field.place(self.mesh.verts, {"K": ti.math.mat3})
        K_edges: ti.MatrixField = self.mesh.edges.get_member_field("K")
        K_edges.fill(0)
        K_verts: ti.MatrixField = self.mesh.verts.get_member_field("K")
        K_verts.fill(0)
        self._stiffness()

    def force(self, disp: npt.ArrayLike) -> npt.NDArray[np.floating]:
        mkit.taichi.mesh.field.place(
            self.mesh.verts, {"f": ti.math.vec3, "u": ti.math.vec3}
        )
        u: ti.MatrixField = self.mesh.verts.get_member_field("u")
        u.from_numpy(np.asarray(disp))
        self._force()
        f_ti: ti.MatrixField = self.mesh.verts.get_member_field("f")
        f_np: npt.NDArray[np.floating] = f_ti.to_numpy()
        return f_np

    def stiffness_sparse_matrix(self) -> scipy.sparse.coo_array:
        builder = ti.linalg.SparseMatrixBuilder(
            self.n_verts * 3, self.n_verts * 3, int(1e7)
        )
        self._stiffness_matrix(builder)
        K_ti: ti.linalg.SparseMatrix = builder.build()
        with tempfile.TemporaryDirectory() as _tmpdir:
            tmpdir = pathlib.Path(_tmpdir)
            file: pathlib.Path = tmpdir / "K.mtx"
            K_ti.mmwrite(str(file))
            K_scipy = scipy.sparse.coo_array(scipy.io.mmread(file))
        return K_scipy

    @no_type_check
    @ti.kernel
    def _volume(self):
        for c in self.mesh.cells:
            dX = ti.Matrix.cols(
                [c.verts[i].pos - c.verts[3].pos for i in ti.static(range(3))]
            )
            c.vol = ti.abs(dX.determinant()) / 6

    @no_type_check
    @ti.kernel
    def _stiffness(self):
        for c in self.mesh.cells:
            M = ti.Matrix.zero(float, 4, 3)
            for i in ti.static(range(4)):
                v = c.verts[i]
                p = [c.verts[(i + j) % 4].pos for j in range(1, 4)]
                Mi = p[0].cross(p[1]) + p[1].cross(p[2]) + p[2].cross(p[0])
                if Mi.dot(v.pos - p[0]) > 0:
                    Mi = -Mi
                M[i, :] = Mi
                v.K += _stiffness_tet(c.vol, Mi, Mi, c.lambda_, c.mu)
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
                e.K += _stiffness_tet(c.vol, Mj, Mk, c.lambda_, c.mu)

    @no_type_check
    @ti.kernel
    def _stiffness_matrix(self, builder: ti.types.sparse_matrix_builder()):
        for v in self.mesh.verts:
            for i, j in ti.static(ti.ndrange(3, 3)):
                row = v.id * 3 + i
                col = v.id * 3 + j
                builder[row, col] += v.K[i, j]
        for e in self.mesh.edges:
            for i, j in ti.static(ti.ndrange(3, 3)):
                row = e.verts[0].id * 3 + i
                col = e.verts[1].id * 3 + j
                builder[row, col] += e.K[i, j]
                builder[col, row] += e.K[j, i]

    @no_type_check
    @ti.kernel
    def _force(self):
        for v in self.mesh.verts:
            v.f = v.K @ v.u
            for e_idx in range(v.edges.size):
                e = v.edges[e_idx]
                if e.verts[0].id == v.id:
                    v.f += e.K @ e.verts[1].u
                elif e.verts[1].id == v.id:
                    v.f += e.K.transpose() @ e.verts[0].u


@no_type_check
@ti.func
def _stiffness_tet(
    vol: float, Mj: ti.math.vec3, Mk: ti.math.vec3, lambda_: float, mu: float
) -> ti.math.mat3:
    return (
        lambda_ * Mk.outer_product(Mj)
        + mu * Mj.outer_product(Mk)
        + mu * Mj.dot(Mk) * ti.math.eye(3)
    ) / (36 * vol)
