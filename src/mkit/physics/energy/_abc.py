import functools
from collections.abc import Callable, Mapping
from typing import Protocol

import jax
import jax.typing as jxt


class EnergyFn(Protocol):
    def __call__(
        self,
        disp: jxt.ArrayLike,
        points: jxt.ArrayLike,
        point_data: Mapping[str, jxt.ArrayLike],
        cell_data: Mapping[str, jxt.ArrayLike],
        field_data: Mapping[str, jxt.ArrayLike],
    ) -> jax.Array: ...


class EnergyFnOptional(Protocol):
    def __call__(
        self,
        disp: jxt.ArrayLike,
        points: jxt.ArrayLike,
        point_data: Mapping[str, jxt.ArrayLike] = {},
        cell_data: Mapping[str, jxt.ArrayLike] = {},
        field_data: Mapping[str, jxt.ArrayLike] = {},
    ) -> jax.Array: ...


def kwargs_to_positional(
    *, vmap: bool = False, jit: bool = True
) -> Callable[[EnergyFn], EnergyFnOptional]:
    def wrapper(fn: EnergyFn) -> EnergyFnOptional:
        if vmap:
            fn = jax.vmap(fn, in_axes=(0, 0, 0, 0, None))
        if jit:
            fn = jax.jit(fn)

        @functools.wraps(fn)
        def wrapped(
            disp: jxt.ArrayLike,
            points: jxt.ArrayLike,
            point_data: Mapping[str, jxt.ArrayLike] = {},
            cell_data: Mapping[str, jxt.ArrayLike] = {},
            field_data: Mapping[str, jxt.ArrayLike] = {},
        ) -> jax.Array:
            return fn(disp, points, point_data, cell_data, field_data)

        return wrapped

    return wrapper


class CellEnergy(EnergyFnOptional):
    fn: EnergyFn

    def __init__(self, fn: "EnergyFn | CellEnergy") -> None:
        if isinstance(fn, CellEnergy):
            self.fn = fn.fn
        else:
            self.fn = fn

    def __add__(self, other: "EnergyFn | CellEnergy") -> "CellEnergy":
        if not isinstance(other, CellEnergy):
            other = CellEnergy(other)
        return CellEnergy(
            lambda *args, **kwargs: self.fn(*args, **kwargs) + other.fn(*args, **kwargs)
        )

    def __call__(
        self,
        disp: jxt.ArrayLike,
        points: jxt.ArrayLike,
        point_data: Mapping[str, jxt.ArrayLike] = {},
        cell_data: Mapping[str, jxt.ArrayLike] = {},
        field_data: Mapping[str, jxt.ArrayLike] = {},
    ) -> jax.Array:
        """(4, 3) -> ()."""
        return self._fn(disp, points, point_data, cell_data, field_data)

    @functools.cached_property
    def jac(self) -> EnergyFnOptional:
        """(4, 3) -> (4, 3)."""
        return kwargs_to_positional()(jax.jacfwd(self.fn))

    @functools.cached_property
    def hess(self) -> EnergyFnOptional:
        """(4, 3) -> (4, 3, 4, 3)."""
        return kwargs_to_positional()(jax.hessian(self.fn))

    @functools.cached_property
    def vmap(self) -> EnergyFnOptional:
        """(C, 4, 3) -> (C,)."""
        return kwargs_to_positional(vmap=True)(self.fn)

    @functools.cached_property
    def jac_vmap(self) -> EnergyFnOptional:
        """(C, 4, 3) -> (C, 4, 3)."""
        return kwargs_to_positional(vmap=True)(jax.jacobian(self.fn))

    @functools.cached_property
    def hess_vmap(self) -> EnergyFnOptional:
        """(C, 4, 3) -> (C, 4, 3, 4, 3)."""
        return kwargs_to_positional(vmap=True)(jax.hessian(self.fn))

    @functools.cached_property
    def _fn(self) -> EnergyFnOptional:
        return kwargs_to_positional()(self.fn)


def cell_energy(fn: EnergyFn) -> CellEnergy:
    return CellEnergy(fn)
