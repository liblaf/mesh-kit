from typing import no_type_check

import scipy.sparse
import taichi as ti

import mkit
import mkit.taichi.mesh.field


def grad(mesh: ti.MeshInstance, *, fp=float, ip=int) -> scipy.sparse.coo_matrix:
    mkit.taichi.mesh.field.place_safe(mesh.cells, {"grad": ti.types.matrix(3, 4, fp)})
    mkit.taichi.mesh.field.place_safe(mesh.verts, {"position": ti.types.vector(3, fp)})
    grad: ti.MatrixField = mesh.cells.get_member_field("grad")
    grad.fill(0)
    position: ti.MatrixField = mesh.verts.get_member_field("position")
    position.from_numpy(mesh.get_position_as_numpy())

    @no_type_check
    @ti.kernel
    def grad_kernel(mesh: ti.template()):
        for c in mesh.cells:
            shape = ti.Matrix.rows(
                [c.verts[i].position - c.verts[3].position for i in ti.static(range(3))]
            )
            shape_inv = shape.inverse()
            c.grad[:, :3] = shape_inv
            c.grad[:, 3] = -shape_inv @ ti.Vector.one(fp, 3)

    grad_kernel(mesh)

    data: ti.ScalarField = ti.field(fp, (12 * len(mesh.cells),))
    row: ti.ScalarField = ti.field(ip, (12 * len(mesh.cells),))
    col: ti.ScalarField = ti.field(ip, (12 * len(mesh.cells),))

    @no_type_check
    @ti.kernel
    def build_matrix(
        mesh: ti.template(), data: ti.template(), row: ti.template(), col: ti.template()
    ):
        for c in mesh.cells:
            for i, j in ti.static(ti.ndrange(3, 4)):
                idx = c.id * 12 + i * 4 + j
                data[idx] = c.grad[i, j]
                row[idx] = c.id * 3 + i
                col[idx] = c.verts[j].id

    build_matrix(mesh, data, row, col)

    return scipy.sparse.coo_matrix((data.to_numpy(), (row.to_numpy(), col.to_numpy())))
