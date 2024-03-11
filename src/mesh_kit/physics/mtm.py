from typing import no_type_check

import taichi as ti

from mesh_kit.taichi import mesh as _mesh


@no_type_check
@ti.kernel
def _calc_stiffness_kernel(mesh: ti.template()):
    for c in mesh.cells:
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
            p = [c.verts[(i + j) % 4].pos for j in range(1, 4)]
            Mi = p[0].cross(p[1]) + p[1].cross(p[2]) + p[2].cross(p[0])
            M[i, :] = Mi
            c.verts[i].K += (
                1
                / (36 * V)
                * (
                    c.lambda_ * Mi.outer_product(Mi)
                    + c.mu * Mi.outer_product(Mi)
                    + c.mu * Mi.dot(Mi) * ti.Matrix.identity(float, 3)
                )
            )
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
            e.K += (
                1
                / (36 * V)
                * (
                    c.lambda_ * Mk.outer_product(Mj)
                    + c.mu * Mj.outer_product(Mk)
                    + c.mu * Mj.dot(Mk) * ti.Matrix.identity(float, 3)
                )
            )


def calc_stiffness(mesh: ti.MeshInstance) -> None:
    assert "lambda_" in mesh.cells.attr_dict
    assert "mu" in mesh.cells.attr_dict
    assert "pos" in mesh.verts.attr_dict
    _mesh.place_safe(mesh.edges, {"K": ti.math.mat3})
    K_edges: ti.MatrixField = mesh.edges.get_member_field("K")
    K_edges.fill(0)
    _mesh.place_safe(mesh.verts, {"K": ti.math.mat3})
    K_verts: ti.MatrixField = mesh.verts.get_member_field("K")
    K_verts.fill(0)
    _calc_stiffness_kernel(mesh)


@no_type_check
@ti.kernel
def _calc_force_kernel(mesh: ti.template()):
    for v in mesh.verts:
        v.f += v.K @ v.x
    for e in mesh.edges:
        e.verts[0].f += e.K @ e.verts[1].x
        e.verts[1].f += e.K.transpose() @ e.verts[0].x


def calc_force(mesh: ti.MeshInstance) -> None:
    assert "K" in mesh.edges.attr_dict
    assert "pos" in mesh.verts.attr_dict
    assert "K" in mesh.verts.attr_dict
    assert "x" in mesh.verts.attr_dict
    _mesh.place_safe(mesh.verts, {"f": ti.math.vec3})
    f: ti.MatrixField = mesh.verts.get_member_field("f")
    f.fill(0)
    _calc_force_kernel(mesh)
