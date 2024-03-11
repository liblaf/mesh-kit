import functools
import pathlib
from typing import Annotated, no_type_check

import meshtaichi_patcher
import numpy as np
import taichi as ti
import trimesh
import typer
from loguru import logger
from mesh_kit import cli, points
from mesh_kit.typing import check_type as _t
from numpy import typing as npt


def init(
    tet_file: pathlib.Path,
    pre_skull_file: pathlib.Path,
    post_skull_file: pathlib.Path,
    *,
    young_modulus: float = 3000,
    poisson_ratio: float = 0.47,
) -> ti.MeshInstance:
    mesh: ti.MeshInstance = meshtaichi_patcher.load_mesh(
        str(tet_file), relations=["CE", "CV", "EV"]
    )
    mesh.cells.place({"lambda_": float, "mu": float})
    # https://en.wikipedia.org/wiki/Template:Elastic_moduli
    E: float = young_modulus
    nu: float = poisson_ratio
    # Lamé's first parameter
    lambda_: ti.ScalarField = mesh.cells.get_member_field("lambda_")
    lambda_.fill(E * nu / ((1 + nu) * (1 - 2 * nu)))
    # Shear modulus
    mu: ti.ScalarField = mesh.cells.get_member_field("mu")
    mu.fill(E / (2 * (1 + nu)))

    mesh.edges.place({"K": ti.math.mat3})
    K_edge: ti.MatrixField = mesh.edges.get_member_field("K")
    K_edge.fill(0)

    mesh.verts.place(
        {
            "pos": ti.math.vec3,
            "fixed": ti.math.vec3,
            "K": ti.math.mat3,
            "x": ti.math.vec3,
            "b": ti.math.vec3,
            "Ax": ti.math.vec3,
        }
    )
    pos: ti.MatrixField = mesh.verts.get_member_field("pos")
    pos_np: npt.NDArray = mesh.get_position_as_numpy()
    pos.from_numpy(pos_np)

    fixed: ti.MatrixField = mesh.verts.get_member_field("fixed")
    pre_skull: trimesh.Trimesh = _t(trimesh.Trimesh, trimesh.load(pre_skull_file))
    post_skull: trimesh.Trimesh = _t(trimesh.Trimesh, trimesh.load(post_skull_file))
    fixed_np: npt.NDArray = np.full((len(mesh.verts), 3), np.nan)
    fixed_idx: npt.NDArray = points.position2idx(
        mesh.get_position_as_numpy(), pre_skull.vertices
    )
    fixed_np[fixed_idx] = post_skull.vertices
    fixed.from_numpy(fixed_np)

    return mesh


@no_type_check
@ti.kernel
def calc_stiffness_kernel(mesh: ti.template()):
    for c in mesh.cells:
        V = ti.abs(
            (
                1
                / 6
                * ti.Matrix.cols(
                    [c.verts[i].pos - c.verts[3].pos for i in range(3)]
                ).determinant()
            )
        )
        M = ti.Matrix.zero(float, 4, 3)
        for i in range(4):
            a = c.verts[(i + 1) % 4].pos
            b = c.verts[(i + 2) % 4].pos
            cc = c.verts[(i + 3) % 4].pos
            M[i, :] = a.cross(b) + b.cross(cc) + cc.cross(a)
            if (c.verts[i].pos - a).dot(M[i, :]) > 0:
                M[i, :] = -M[i, :]
            c.verts[i].K += (
                1
                / (36 * V)
                * (
                    c.lambda_ * M[i, :].outer_product(M[i, :])
                    + c.mu * M[i, :].outer_product(M[i, :])
                    + c.mu * M[i, :].dot(M[i, :]) * ti.Matrix.identity(float, 3)
                )
            )
        for e_idx in range(6):
            e = c.edges[e_idx]
            v_idx = ti.Vector.zero(int, 2)
            for i in range(2):
                for j in range(4):
                    if e.verts[i].id == c.verts[j].id:
                        v_idx[i] = j
                        break
            j, k = v_idx
            e.K += (
                1
                / (36 * V)
                * (
                    c.lambda_ * M[k, :].outer_product(M[j, :])
                    + c.mu * M[j, :].outer_product(M[k, :])
                    + c.mu * M[j, :].dot(M[k, :]) * ti.Matrix.identity(float, 3)
                )
            )


def calc_stiffness(mesh: ti.MeshInstance) -> None:
    K_edges: ti.MatrixField = mesh.edges.get_member_field("K")
    K_edges.fill(0)
    K_verts: ti.MatrixField = mesh.verts.get_member_field("K")
    K_verts.fill(0)
    calc_stiffness_kernel(mesh)


@no_type_check
@ti.func
def is_fixed(fixed: ti.math.vec3) -> bool:
    return not ti.math.isnan(fixed).any()


@no_type_check
@ti.kernel
def calc_b_kernel(mesh: ti.template()):
    """b = - K10 @ x0"""
    for v in mesh.verts:
        if is_fixed(v.fixed):
            v.x = v.fixed - v.pos
            # v.b = v.fixed - v.pos
        else:
            v.b = 0
    # for e in mesh.edges:
    #     if not is_fixed(e.verts[0].fixed) and is_fixed(e.verts[1].fixed):
    #         e.verts[0].b -= e.K @ e.verts[1].x
    #     elif is_fixed(e.verts[0].fixed) and not is_fixed(e.verts[1].fixed):
    #         e.verts[1].b -= e.K.transpose() @ e.verts[0].x


def calc_b(mesh: ti.MeshInstance) -> None:
    b: ti.MatrixField = mesh.verts.get_member_field("b")
    b.fill(0)
    calc_b_kernel(mesh)


@no_type_check
@ti.kernel
def calc_Ax_kernel(mesh: ti.template(), x: ti.template(), Ax: ti.template()):
    for v in mesh.verts:
        v.x = x[v.id]
        v.Ax = ti.Vector.zero(float, 3)
    for v in mesh.verts:
        if is_fixed(v.fixed):
            v.Ax = 0
        else:
            v.Ax += v.K @ v.x
    for e in mesh.edges:
        if not is_fixed(e.verts[0].fixed) and not is_fixed(e.verts[1].fixed):
            e.verts[0].Ax += e.K @ e.verts[1].x
            e.verts[1].Ax += e.K.transpose() @ e.verts[0].x
    for v in mesh.verts:
        Ax[v.id] = v.Ax


def calc_Ax(mesh: ti.MeshInstance, x: ti.MatrixField, Ax: ti.MatrixField) -> None:
    """Ax = K11 @ x1"""
    calc_Ax_kernel(mesh, x, Ax)


def MatrixFreeCG(
    A: ti.linalg.LinearOperator,
    b: ti.MatrixField,
    x: ti.MatrixField,
    tol=1e-6,
    maxiter=5000,
    quiet=True,
) -> bool:
    """Matrix-free conjugate-gradient solver.

    Use conjugate-gradient method to solve the linear system Ax = b, where A is implicitly
    represented as a LinearOperator.

    Args:
        A (LinearOperator): The coefficient matrix A of the linear system.
        b (Field): The right-hand side of the linear system.
        x (Field): The initial guess for the solution.
        maxiter (int): Maximum number of iterations.
        atol: Tolerance(absolute) for convergence.
        quiet (bool): Switch to turn on/off iteration log.
    """

    if b.dtype != x.dtype:
        raise ti.TaichiTypeError(
            f"Dtype mismatch b.dtype({b.dtype}) != x.dtype({x.dtype})."
        )
    if str(b.dtype) == "f32":
        solver_dtype = ti.f32
    elif str(b.dtype) == "f64":
        solver_dtype = ti.f64
    else:
        raise ti.TaichiTypeError(f"Not supported dtype: {b.dtype}")
    if b.shape != x.shape:
        raise ti.TaichiRuntimeError(
            f"Dimension mismatch b.shape{b.shape} != x.shape{x.shape}."
        )

    size = b.shape
    vector_fields_builder = ti.FieldsBuilder()
    p: ti.MatrixField = ti.Vector.field(3, dtype=solver_dtype)
    r: ti.MatrixField = ti.Vector.field(3, dtype=solver_dtype)
    Ap: ti.MatrixField = ti.Vector.field(3, dtype=solver_dtype)
    Ax: ti.MatrixField = ti.Vector.field(3, dtype=solver_dtype)
    if len(size) == 1:
        axes = ti.i
    elif len(size) == 2:
        axes = ti.ij
    elif len(size) == 3:
        axes = ti.ijk
    else:
        raise ti.TaichiRuntimeError(
            f"MatrixFreeCG only support 1D, 2D, 3D inputs; your inputs is {len(size)}-D."
        )
    vector_fields_builder.dense(axes, size).place(p, r, Ap, Ax)
    vector_fields_snode_tree = vector_fields_builder.finalize()

    scalar_builder = ti.FieldsBuilder()
    alpha = ti.field(dtype=solver_dtype)
    beta = ti.field(dtype=solver_dtype)
    scalar_builder.place(alpha, beta)
    scalar_snode_tree = scalar_builder.finalize()
    success: bool = True

    @no_type_check
    @ti.kernel
    def init():
        for I in ti.grouped(x):
            r[I] = b[I] - Ax[I]
            p[I] = 0.0
            Ap[I] = 0.0

    @no_type_check
    @ti.kernel
    def reduce(p: ti.template(), q: ti.template()) -> solver_dtype:
        result = solver_dtype(0.0)
        for I in ti.grouped(p):
            result += p[I].dot(q[I])
        return result

    @no_type_check
    @ti.kernel
    def update_x():
        for I in ti.grouped(x):
            x[I] += alpha[None] * p[I]

    @no_type_check
    @ti.kernel
    def update_r():
        for I in ti.grouped(r):
            r[I] -= alpha[None] * Ap[I]

    @no_type_check
    @ti.kernel
    def update_p():
        for I in ti.grouped(p):
            p[I] = r[I] + beta[None] * p[I]

    def solve() -> bool:
        A._matvec(x, Ax)
        init()
        initial_rTr = reduce(r, r)
        if not quiet:
            print(f">>> Initial residual = {initial_rTr:e}")
        old_rTr = initial_rTr
        new_rTr = initial_rTr
        update_p()
        if ti.sqrt(initial_rTr) >= tol:
            # Do nothing if the initial residual is small enough
            # -- Main loop --
            for i in range(maxiter):
                A._matvec(p, Ap)  # compute Ap = A x p
                pAp = reduce(p, Ap)
                alpha[None] = old_rTr / pAp
                update_x()
                update_r()
                new_rTr = reduce(r, r)
                if ti.sqrt(new_rTr) < tol:
                    logger.debug("Conjugate Gradient method converged.")
                    logger.debug("#iterations {}", i)
                    break
                beta[None] = new_rTr / old_rTr
                update_p()
                old_rTr = new_rTr
                logger.debug("Iter = {:4}, Residual = {:e}", i + 1, ti.sqrt(new_rTr))
        if new_rTr >= tol:
            logger.debug(
                "Conjugate Gradient method failed to converge in {} iterations: Residual = {:e}",
                maxiter,
                ti.sqrt(new_rTr),
            )
            success = False

    success: bool = solve()
    vector_fields_snode_tree.destroy()
    scalar_snode_tree.destroy()
    return success


def main(
    tet_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    pre_skull_file: Annotated[
        pathlib.Path, typer.Option("--pre", exists=True, dir_okay=False)
    ],
    post_skull_file: Annotated[
        pathlib.Path, typer.Option("--post", exists=True, dir_okay=False)
    ],
    young_modulus: Annotated[float, typer.Option(min=0)] = 3000,
    poisson_ratio: Annotated[float, typer.Option(min=0, max=0.5)] = 0.47,
) -> None:
    ti.init()
    mesh: ti.MeshInstance = init(
        tet_file,
        pre_skull_file,
        post_skull_file,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
    )
    calc_stiffness(mesh)
    A = ti.linalg.LinearOperator(functools.partial(calc_Ax, mesh))
    calc_b(mesh)
    b: ti.MatrixField = mesh.verts.get_member_field("b")
    x: ti.MatrixField = mesh.verts.get_member_field("x")
    x.fill(0)
    success: bool = MatrixFreeCG(A, b, x, quiet=False)
    print("success =", success)


if __name__ == "__main__":
    cli.run(main)
