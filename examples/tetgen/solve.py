import functools
import math
import pathlib
from typing import Annotated, Any, no_type_check

import meshtaichi_patcher
import numpy as np
import taichi as ti
import trimesh
import typer
from loguru import logger
from mesh_kit import cli, points
from mesh_kit.io import node
from mesh_kit.physics import mtm
from mesh_kit.typing import check_type as _t
from numpy import typing as npt
from taichi.lang.exception import TaichiRuntimeError, TaichiTypeError


def init(
    tet_file: pathlib.Path,
    pre_skull_file: pathlib.Path,
    post_skull_file: pathlib.Path,
    *,
    young_modulus: float = 3000,
    poisson_ratio: float = 0.47,
) -> ti.MeshInstance:
    mesh: ti.MeshInstance = meshtaichi_patcher.load_mesh(
        str(tet_file), relations=["CE", "CV", "EV", "VE"]
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
@ti.func
def is_fixed(fixed: ti.math.vec3) -> bool:
    return not ti.math.isnan(fixed).any()


@no_type_check
@ti.kernel
def calc_b_kernel(mesh: ti.template()):
    """b = - K10 @ x0"""
    for v in mesh.verts:
        if not is_fixed(v.fixed):
            for e_idx in range(v.edges.size):
                e = v.edges[e_idx]
                if e.verts[0].id == v.id and is_fixed(e.verts[1].fixed):
                    v.b -= e.K @ (e.verts[1].fixed - e.verts[1].pos)
                elif e.verts[1].id == v.id and is_fixed(e.verts[0].fixed):
                    v.b -= e.K.transpose() @ (e.verts[0].fixed - e.verts[0].pos)


def calc_b(mesh: ti.MeshInstance) -> None:
    """b = - K10 @ x0"""
    b: ti.MatrixField = mesh.verts.get_member_field("b")
    b.fill(0)
    calc_b_kernel(mesh)


@no_type_check
@ti.kernel
def calc_Ax_kernel(mesh: ti.template(), x: ti.template(), Ax: ti.template()):
    """Ax = K11 @ x1"""
    for v in mesh.verts:
        # if is_fixed(v.fixed):
        #     v.x = v.fixed - v.pos
        # else:
        #     v.x = x[v.id]
        v.x = x[v.id]
        v.Ax = 0
    for v in mesh.verts:
        if not is_fixed(v.fixed):
            v.Ax += v.K @ v.x
            for e_idx in range(v.edges.size):
                e = v.edges[e_idx]
                if e.verts[0].id == v.id and not is_fixed(e.verts[1].fixed):
                    v.Ax += e.K @ e.verts[1].x
                elif e.verts[1].id == v.id and not is_fixed(e.verts[0].fixed):
                    v.Ax += e.K.transpose() @ e.verts[0].x
    for v in mesh.verts:
        Ax[v.id] = v.Ax


def calc_Ax(mesh: ti.MeshInstance, x: ti.MatrixField, Ax: ti.MatrixField) -> None:
    """Ax = K11 @ x1"""
    calc_Ax_kernel(mesh, x, Ax)


def MatrixFreeCG(
    A: ti.linalg.LinearOperator,
    b: ti.MatrixField,
    x: ti.MatrixField,
    mesh: ti.MeshInstance,
    tol: float = 1e-6,
    maxiter: int = 5000,
    quiet: bool = True,
) -> bool:
    """Matrix-free conjugate-gradient solver.

    Use conjugate-gradient method to solve the linear system Ax = b, where A is implicitly
    represented as a LinearOperator.

    Args:
        A (LinearOperator): The coefficient matrix A of the linear system.
        b (MatrixField): The right-hand side of the linear system.
        x (MatrixField): The initial guess for the solution.
        maxiter (int): Maximum number of iterations.
        atol: Tolerance(absolute) for convergence.
        quiet (bool): Switch to turn on/off iteration log.
    """  # noqa: E501
    if b.dtype != x.dtype:
        raise TaichiTypeError(
            f"Dtype mismatch b.dtype({b.dtype}) != x.dtype({x.dtype})."
        )
    solver_dtype: Any
    match str(b.dtype):
        case "f32":
            solver_dtype = ti.f32
        case "f64":
            solver_dtype = ti.f64
        case _:
            raise TaichiTypeError(f"Not supported dtype: {b.dtype}")
    if b.shape != x.shape:
        raise TaichiRuntimeError(
            f"Dimension mismatch b.shape{b.shape} != x.shape{x.shape}."
        )

    size: tuple[int, ...] = b.shape
    vector_fields_builder = ti.FieldsBuilder()
    p: ti.MatrixField = ti.Vector.field(n=b.n, dtype=b.dtype)
    r: ti.MatrixField = ti.Vector.field(n=b.n, dtype=b.dtype)
    Ap: ti.MatrixField = ti.Vector.field(n=b.n, dtype=b.dtype)
    Ax: ti.MatrixField = ti.Vector.field(n=b.n, dtype=b.dtype)
    axes: list
    match len(size):
        case 1:
            axes = ti.i
        case 2:
            axes = ti.ij
        case 3:
            axes = ti.ijk
        case _:
            raise TaichiRuntimeError(
                f"MatrixFreeCG only support 1D, 2D, 3D inputs; your inputs is {len(size)}-D."  # noqa: E501
            )
    vector_fields_builder.dense(axes, size).place(p, r, Ap, Ax)
    vector_fields_snode_tree = vector_fields_builder.finalize()

    scalar_builder = ti.FieldsBuilder()
    alpha: ti.ScalarField = ti.field(b.dtype)
    beta: ti.ScalarField = ti.field(b.dtype)
    scalar_builder.place(alpha, beta)
    scalar_snode_tree = scalar_builder.finalize()

    @no_type_check
    @ti.kernel
    def init():
        for I in ti.grouped(x):  # noqa: E741
            r[I] = b[I] - Ax[I]
            p[I] = 0.0
            Ap[I] = 0.0

    @no_type_check
    @ti.kernel
    def reduce(p: ti.template(), q: ti.template()) -> solver_dtype:
        result = solver_dtype(0.0)
        for v in mesh.verts:
            if not is_fixed(v.fixed):
                result += p[v.id].dot(q[v.id])
        return result
        # result = solver_dtype(0.0)
        # for I in ti.grouped(p):  # noqa: E741
        #     result += p[I].dot(q[I])
        # return result

    @no_type_check
    @ti.kernel
    def update_x():
        for I in ti.grouped(x):  # noqa: E741
            x[I] += alpha[None] * p[I]

    @no_type_check
    @ti.kernel
    def update_r():
        for I in ti.grouped(r):  # noqa: E741
            r[I] -= alpha[None] * Ap[I]

    @no_type_check
    @ti.kernel
    def update_p():
        for I in ti.grouped(p):  # noqa: E741
            p[I] = r[I] + beta[None] * p[I]

    def solve() -> bool:
        succeeded: bool = True
        A._matvec(x, Ax)
        init()
        initial_rTr: float = reduce(r, r)
        if not quiet:
            logger.info("Initial residual = {:e}", initial_rTr)
        old_rTr: float = initial_rTr
        new_rTr: float = initial_rTr
        update_p()
        if math.sqrt(initial_rTr) >= tol:
            # Do nothing if the initial residual is small enough
            # -- Main loop --
            for i in range(maxiter):
                A._matvec(p, Ap)  # compute Ap = A x p
                pAp: float = reduce(p, Ap)
                alpha[None] = old_rTr / pAp
                update_x()
                update_r()
                new_rTr = reduce(r, r)
                if not quiet:
                    logger.debug(
                        "Iter = {:4}, Residual = {:e}", i + 1, math.sqrt(new_rTr)
                    )
                if math.sqrt(new_rTr) < tol:
                    if not quiet:
                        logger.info("Conjugate Gradient method converged.")
                        logger.info("#iterations {}", i)
                    break
                beta[None] = new_rTr / old_rTr
                update_p()
                old_rTr = new_rTr
        if math.sqrt(new_rTr) >= tol:
            if not quiet:
                logger.warning(
                    "Conjugate Gradient method failed to converge in {} iterations: Residual = {:e}",  # noqa: E501
                    maxiter,
                    math.sqrt(new_rTr),
                )
            succeeded = False
        return succeeded

    succeeded: bool = solve()
    vector_fields_snode_tree.destroy()
    scalar_snode_tree.destroy()
    return succeeded


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
    ti.init(default_fp=ti.f64)
    mesh: ti.MeshInstance = init(
        tet_file,
        pre_skull_file,
        post_skull_file,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
    )
    mtm.calc_stiffness(mesh)
    A = ti.linalg.LinearOperator(functools.partial(calc_Ax, mesh))
    calc_b(mesh)
    b: ti.MatrixField = mesh.verts.get_member_field("b")
    x: ti.MatrixField = mesh.verts.get_member_field("x")
    x.fill(0)
    success: bool = MatrixFreeCG(A, b, x, mesh, tol=1e-10, maxiter=50000, quiet=False)
    mtm.calc_force(mesh)
    Ax: ti.MatrixField = mesh.verts.get_member_field("Ax")
    print("b_max =", np.abs(b.to_numpy()).max())
    print("x_max =", np.abs(x.to_numpy()).max())
    print("Ax_max =", np.abs(Ax.to_numpy()).max())
    print(np.sum(np.abs(Ax.to_numpy() - b.to_numpy())))
    print("f_max =", np.abs(mesh.verts.get_member_field("f").to_numpy()).max())
    assert success
    node.save(
        pathlib.Path("data") / "post.1.node",
        mesh.get_position_as_numpy() + x.to_numpy(),
        comment=False,
    )


if __name__ == "__main__":
    cli.run(main)
