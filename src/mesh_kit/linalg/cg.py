# https://github.com/taichi-dev/taichi/blob/master/python/taichi/linalg/matrixfree_cg.py
from math import sqrt
from typing import Any, no_type_check

import taichi as ti
from loguru import logger
from taichi.lang.exception import TaichiRuntimeError, TaichiTypeError


def MatrixFreeCG(
    A: ti.linalg.LinearOperator,
    b: ti.MatrixField,
    x: ti.MatrixField,
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
        for I in ti.grouped(p):  # noqa: E741
            result += p[I].dot(q[I])
        return result

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
        if sqrt(initial_rTr) >= tol:
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
                    logger.debug("Iter = {:4}, Residual = {:e}", i + 1, sqrt(new_rTr))
                if sqrt(new_rTr) < tol:
                    if not quiet:
                        logger.info("Conjugate Gradient method converged.")
                        logger.info("#iterations {}", i)
                    break
                beta[None] = new_rTr / old_rTr
                update_p()
                old_rTr = new_rTr
        if sqrt(new_rTr) >= tol:
            if not quiet:
                logger.warning(
                    "Conjugate Gradient method failed to converge in {} iterations: Residual = {:e}",  # noqa: E501
                    maxiter,
                    sqrt(new_rTr),
                )
            succeeded = False
        return succeeded

    succeeded: bool = solve()
    vector_fields_snode_tree.destroy()
    scalar_snode_tree.destroy()
    return succeeded
