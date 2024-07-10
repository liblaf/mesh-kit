import abc
import functools
from collections.abc import Callable, Iterable
from typing import Any, ParamSpec, TypeVar

import jax
import numpy as np
import numpy.typing as npt
import taichi as ti

import mkit.io
import mkit.physics.common.element


class Model:
    mesh: ti.MeshInstance
    tetra: jax.Array
    energy_element: Callable

    def __init__(self, mesh: Any, relations: Iterable[str] | None = None) -> None:
        if relations is None:
            relations = ["CV"]
        self.mesh = mkit.io.as_taichi(mesh, relations)

    @abc.abstractmethod
    def calc_energy(self) -> jax.Array: ...

    @property
    def n_cells(self) -> int:
        return len(self.mesh.cells)

    @property
    def n_verts(self) -> int:
        return len(self.mesh.verts)

    @property
    def position(self) -> ti.MatrixField:
        return self.mesh.verts.get_member_field("position")

    @functools.cached_property
    def position_rest(self) -> npt.NDArray[np.floating]:
        return self.mesh.get_position_as_numpy()

    @functools.cached_property
    def volume(self) -> jax.Array:
        return jax.vmap(mkit.physics.common.element.calc_volume)(
            self.position_rest[self.tetra]
        )


P = ParamSpec("P")
T = TypeVar("T")


def energy_element_jac(func: Callable[P, T]) -> Callable[P, jax.Array]:
    return jax.jacobian(func)  # pyright: ignore [reportReturnType]


def energy_element_hess(func: Callable[P, T]) -> Callable[P, jax.Array]:
    return jax.hessian(func)  # pyright: ignore [reportReturnType]
