from typing import no_type_check

import taichi as ti

from mkit.physics import moduli as _moduli


@ti.data_oriented
class MTM:
    """Mass-Tensor Model"""

    mesh: ti.MeshInstance

    def __init__(self, mesh: ti.MeshInstance) -> None:
        self.mesh = mesh

    def init_material(self, E: float, nu: float) -> None:
        """
        Args:
            E: Young's modulus
            nu: Poisson's ratio
        """
        lambda_: float = _moduli.E_nu2lambda(E, nu)
        lambda_ti: ti.ScalarField = self.mesh.cells.get_member_field("lambda_")
        lambda_ti.fill(lambda_)
        mu: float = _moduli.E_nu2mu(E, nu)
        mu_ti: ti.ScalarField = self.mesh.cells.get_member_field("mu")
        mu_ti.fill(mu)

    def volume(self) -> ti.ScalarField:
        """
        Args:
            verts.pos (ti.math.vec3): rest position of vertex

        Returns:
            cells.volume (ti.f64): volume of cell
        """
        self._volume()
        volume: ti.ScalarField = self.mesh.cells.get_member_field("volume")
        return volume

    def stiffness_matrix(self) -> tuple[ti.MatrixField, ti.MatrixField]:
        """
        Args:
            cells.volume (ti.f64): volume of cell
            verts.pos (ti.math.vec3): rest position of vertex
            cells.lambda_ (ti.f64): Lamé's first parameter
            cells.mu (ti.f64): Lamé's second parameter / Shear modulus

        Returns:
            edges.K (ti.math.vec3): stiffness matrix
            verts.K (ti.math.vec3): stiffness matrix
        """
        K_edges: ti.MatrixField = self.mesh.edges.get_member_field("K")
        K_edges.fill(0)
        K_verts: ti.MatrixField = self.mesh.verts.get_member_field("K")
        K_verts.fill(0)
        self._stiffness_matrix()
        return K_edges, K_verts

    def deformation_gradient(self) -> ti.MatrixField:
        """
        Args:
            verts.pos (ti.math.vec3): rest position of vertex
            verts.u (ti.math.vec3): absolute displacement of vertex

        Returns:
            cells.F (ti.math.mat3): deformation gradient
        """
        self._deformation_gradient()
        F: ti.MatrixField = self.mesh.cells.get_member_field("F")
        return F

    def deformation_gradient_grad(self) -> ti.MatrixField:
        """
        Args:
            verts.pos (ti.math.vec3): rest position of vertex
            cells.F_grad (ti.math.mat3): grad of deformation gradient

        Returns:
            verts.u_grad (ti.math.vec3): grad of absolute displacement of vertex
        """
        u_grad: ti.MatrixField = self.mesh.verts.get_member_field("u_grad")
        u_grad.fill(0)
        self._deformation_gradient_grad()
        return u_grad

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
    def _stiffness_matrix(self):
        for c in self.mesh.cells:
            M = ti.Matrix.zero(ti.f64, 4, 3)
            for i in ti.static(range(4)):
                v = c.verts[i]
                p = [c.verts[(i + j) % 4].pos for j in range(1, 4)]
                Mi = p[0].cross(p[1]) + p[1].cross(p[2]) + p[2].cross(p[0])
                if Mi.dot(v.pos - p[0]) > 0:
                    Mi = -Mi
                M[i, :] = Mi
                v.K += _stiffness_matrix_tet(c.volume, Mi, Mi, c.lambda_, c.mu)
            for e_id in ti.static(range(6)):
                e = c.edges[e_id]
                v_idx = ti.Vector.zero(int, 2)  # vertex index in cell (in range(3))
                for i in ti.static(range(2)):
                    for j in range(4):
                        if e.verts[i].id == c.verts[j].id:
                            v_idx[i] = j
                            break
                j, k = v_idx
                Mj, Mk = M[j, :], M[k, :]
                e.K += _stiffness_matrix_tet(c.volume, Mj, Mk, c.lambda_, c.mu)

    @no_type_check
    def _deformation_gradient(self):
        for c in self.mesh.cells:
            dX = ti.Matrix.cols(
                [c.verts[i].pos - c.verts[3].pos for i in ti.static(range(3))]
            )
            x = [c.verts[i].pos + c.verts[i].u for i in ti.static(range(4))]
            dx = ti.Matrix.cols([x[i] - x[3] for i in ti.static(range(3))])
            c.F = dx @ dX.inverse()

    @no_type_check
    @ti.kernel
    def _deformation_gradient_grad(self):
        for c in self.mesh.cells:
            dX = ti.Matrix.cols(
                [c.verts[i].pos - c.verts[3].pos for i in ti.static(range(3))]
            )
            grad = c.F_grad @ dX.inverse().transpose()
            c.verts[0].u_grad += grad[:, 0]
            c.verts[1].u_grad += grad[:, 1]
            c.verts[2].u_grad += grad[:, 2]
            c.verts[3].u_grad -= grad[:, 0] + grad[:, 1] + grad[:, 2]


@no_type_check
@ti.func
def _stiffness_matrix_tet(
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
