# https://github.com/taichi-dev/taichi/blob/master/python/taichi/linalg/matrixfree_cg.py
import math
from typing import Any, no_type_check

import taichi as ti
from loguru import logger


def get_field_dtype(field: ti.Field) -> Any:
    match field:
        case ti.ScalarField():
            return field.dtype
        case ti.MatrixField():
            match field.ndim:
                case 1:
                    return ti.types.vector(n=field.n, dtype=field.dtype)
                case 2:
                    return ti.types.matrix(n=field.n, m=field.m, dtype=field.dtype)
                case _:
                    raise ValueError(f"Unsupported field.ndim: {field.ndim}")
        case _:
            raise ValueError(f"Unsupported field: {type(field)}")


@no_type_check
@ti.kernel
def _add(f: ti.template(), a: float, x: ti.template(), b: ti.template()):
    for I in ti.grouped(f):  # noqa: E741
        f[I] = a * x[I] + b[I]


@no_type_check
@ti.kernel
def _dot_scalar(a: ti.template(), b: ti.template()) -> float:
    result = 0.0
    for I in ti.grouped(a):  # noqa: E741
        result += a[I] * b[I]
    return result


@no_type_check
@ti.kernel
def _dot_vector(a: ti.template(), b: ti.template()) -> float:
    result = 0.0
    for I in ti.grouped(a):  # noqa: E741
        result += a[I].dot(b[I])
    return result


def _dot(a: ti.Field, b: ti.Field) -> float:
    match a:
        case ti.ScalarField():
            return _dot_scalar(a, b)
        case ti.MatrixField():
            match a.ndim:
                case 1:
                    return _dot_vector(a, b)
                case _:
                    raise ValueError(f"Unsupported field.ndim: {a.ndim}")
        case _:
            raise ValueError(f"Unsupported field: {type(a)}")


def cg(
    A: ti.linalg.LinearOperator,
    b: ti.Field,
    x: ti.Field,
    tol: float = 1e-6,
    maxiter: int = 5000,
) -> bool:
    """https://en.wikipedia.org/wiki/Conjugate_gradient_method"""
    dtype: Any = get_field_dtype(b)
    shape: tuple[int, ...] = b.shape
    Ap: ti.Field = ti.field(dtype=dtype, shape=shape)
    p: ti.Field = ti.field(dtype=dtype, shape=shape)
    r: ti.Field = ti.field(dtype=dtype, shape=shape)
    # Ap = A x
    A.matvec(x, Ap)
    # r = b - A x
    _add(r, -1, Ap, b)
    rTr: float = _dot(r, r)
    if math.sqrt(rTr) < tol:
        logger.success("CG converged in 0 iters: residual = {}", math.sqrt(rTr))
        return True
    logger.debug("iter = 0, residual = {}", math.sqrt(rTr))
    p.copy_from(r)
    for iter in range(maxiter):
        A.matvec(p, Ap)
        # alpha = r^T r / p^T A p
        alpha: float = rTr / _dot(p, Ap)
        # x = x + alpha * p
        _add(x, alpha, p, x)
        # r = r - alpha * A p
        _add(r, -alpha, Ap, r)
        rTr_old: float = rTr
        rTr = _dot(r, r)
        if math.sqrt(rTr) < tol:
            logger.success(
                "CG converged in {} iters: residual = {}", iter + 1, math.sqrt(rTr)
            )
            break
        logger.debug("iter = {}, residual = {}", iter + 1, math.sqrt(rTr))
        beta: float = rTr / rTr_old
        # p = r + beta * p
        _add(p, beta, p, r)
    else:
        logger.warning(
            "CG failed to converge in {} iters: residual = {}", maxiter, math.sqrt(rTr)
        )
    return math.sqrt(rTr) < tol
