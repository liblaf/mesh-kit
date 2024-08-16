import functools
from collections.abc import Mapping
from typing import Protocol

import jax
import jax.typing as jxt


class EnergyFn(Protocol):
    def __call__(
        self,
        disp: jxt.ArrayLike,
        points: jxt.ArrayLike,
        point_data: Mapping[str, jxt.ArrayLike] = {},
        cell_data: Mapping[str, jxt.ArrayLike] = {},
        field_data: Mapping[str, jxt.ArrayLike] = {},
    ) -> jax.Array: ...


class CellEnergyFn:
    fn: EnergyFn

    def __init__(self, fn: EnergyFn) -> None:
        self.fn = fn

    def __add__(self, other: "EnergyFn | CellEnergyFn") -> "CellEnergyFn":
        if not isinstance(other, CellEnergyFn):
            other = CellEnergyFn(other)
        return CellEnergyFn(
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
        return self.fn(
            disp,
            points,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
        )

    @functools.cached_property
    def jac(self) -> EnergyFn:
        """(4, 3) -> (4, 3)."""
        return jax.jacobian(self.fn)

    @functools.cached_property
    def hess(self) -> EnergyFn:
        """(4, 3) -> (4, 3, 4, 3)."""
        return jax.hessian(self.fn)

    @functools.cached_property
    def vmap(self) -> EnergyFn:
        """(C, 4, 3) -> (C,)."""
        return jax.vmap(self.fn)

    @functools.cached_property
    def vmap_jac(self) -> EnergyFn:
        """(C, 4, 3) -> (C, 4, 3)."""
        return jax.vmap(self.jac)

    @functools.cached_property
    def vmap_hess(self) -> EnergyFn:
        """(C, 4, 3) -> (C, 4, 3, 4, 3)."""
        return jax.vmap(self.hess)
