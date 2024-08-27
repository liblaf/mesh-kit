from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.typing as jxt
import matplotlib as mpl
import matplotlib.pyplot as plt
import mkit.creation.tetmesh
from icecream import ic
from mkit.physics.energy.abc import CellEnergy
from mkit.physics.preset.elastic import MODELS

if TYPE_CHECKING:
    import pyvista as pv


def E_nu_to_lambda_mu(E: float, nu: float) -> tuple[float, float]:  # noqa: N802, N803
    lambda_: float = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu: float = E / (2 * (1 + nu))
    return lambda_, mu


def compute_energy(
    x: jxt.ArrayLike,
    energy_fn: CellEnergy,
    points: jxt.ArrayLike,
    cell_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    def single(x: jxt.ArrayLike) -> jax.Array:
        disp: jax.Array = jnp.zeros((4, 3))
        disp = disp.at[0, 0].set(x)
        W: jax.Array = energy_fn(disp, points, cell_data=cell_data)
        return W

    return jax.vmap(single)(x) - single(0)


def compute_energy_jac_0_x(
    x: jxt.ArrayLike,
    energy_fn: CellEnergy,
    points: jxt.ArrayLike,
    cell_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    def single(x: jxt.ArrayLike) -> jax.Array:
        disp: jax.Array = jnp.zeros((4, 3))
        disp = disp.at[0, 0].set(x)
        jac: jax.Array = energy_fn.jac(disp, points, cell_data=cell_data)
        return jac[0, 0]

    return jax.vmap(single)(x)


def compute_energy_jac_2_z(
    x: jxt.ArrayLike,
    energy_fn: CellEnergy,
    points: jxt.ArrayLike,
    cell_data: Mapping[str, jxt.ArrayLike],
) -> jax.Array:
    def single(x: jxt.ArrayLike) -> jax.Array:
        disp: jax.Array = jnp.zeros((4, 3))
        disp = disp.at[0, 0].set(x)
        jac: jax.Array = energy_fn.jac(disp, points, cell_data=cell_data)
        return jac[2, 2]

    return jax.vmap(single)(x)


def main() -> None:
    mpl.rcParams["figure.dpi"] = 300
    x: jax.Array = jnp.linspace(-1, 2)
    mesh: pv.UnstructuredGrid = mkit.creation.tetmesh.tetrahedron()
    ic(mesh.points)

    plt.figure()
    for cfg in MODELS.values():
        ic(cfg.name, cfg.params)
        W: jax.Array = compute_energy(x, cfg.energy_fn, mesh.points, cfg.params)
        energy: jax.Array = mesh.volume * W
        plt.plot(x, energy, label=cfg.name)
    plt.legend()
    plt.xlabel("Displacement")
    plt.ylabel("Energy")
    plt.ylim(top=1e7)
    plt.savefig("plot/curve/energy.svg")
    plt.close()

    plt.figure()
    for cfg in MODELS.values():
        jac: jax.Array = compute_energy_jac_0_x(
            x, cfg.energy_fn, mesh.points, cfg.params
        )
        f: jax.Array = mesh.volume * jac
        plt.plot(x, f, label=cfg.name)
    plt.legend()
    plt.xlabel("Displacement")
    plt.ylabel("Normal Force")
    plt.ylim(-1e7, 1e7)
    plt.savefig("plot/curve/force-normal.svg")
    plt.close()

    plt.figure()
    for cfg in MODELS.values():
        jac: jax.Array = compute_energy_jac_2_z(
            x, cfg.energy_fn, mesh.points, cfg.params
        )
        f: jax.Array = mesh.volume * jac
        plt.plot(x, f, label=cfg.name)
    plt.legend()
    plt.xlabel("Displacement")
    plt.ylabel("Shear Force")
    plt.ylim(top=1e7)
    plt.savefig("plot/curve/force-shear.svg")
    plt.close()


if __name__ == "__main__":
    main()
