import functools
import pathlib
from typing import Annotated

import meshio
import mkit.cli
import mkit.io
import mkit.physics.moduli
import mkit.physics.mtm
import numpy as np
import scipy.sparse.linalg
import taichi as ti
import typer
from loguru import logger
from numpy import typing as npt


@ti.data_oriented
class Model(mkit.physics.mtm.MTM):
    disp: npt.NDArray[np.floating]

    def __init__(self, mesh: meshio.Mesh) -> None:
        super().__init__(mesh, ["CE", "CV", "EV", "VE"])
        self.disp = np.asarray(mesh.point_data["disp"])

    @functools.cached_property
    def fixed_mask(self) -> npt.NDArray[np.bool_]:
        return ~self.free_mask

    @functools.cached_property
    def free_mask(self) -> npt.NDArray[np.bool_]:
        fixed_mask: npt.NDArray[np.bool_] = np.isnan(self.disp).any(axis=1)
        return fixed_mask

    @functools.cached_property
    def n_free(self) -> int:
        return np.count_nonzero(self.free_mask)

    def force_free(self, u_free: npt.ArrayLike) -> npt.NDArray[np.floating]:
        u: npt.NDArray[np.floating] = self.disp.copy()
        u[self.free_mask] = u_free
        f: npt.NDArray[np.floating] = self.force(u)
        return f

    def Ax(self, u_free: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        u: npt.NDArray[np.floating] = np.zeros((self.n_verts, 3))
        u[self.free_mask] = u_free
        f: npt.NDArray[np.floating] = self.force(u)
        return f[self.free_mask]

    def b(self) -> npt.NDArray[np.floating]:
        u: npt.NDArray[np.floating] = np.zeros((self.n_verts, 3))
        u[self.fixed_mask] = self.disp[self.fixed_mask]
        f: npt.NDArray[np.floating] = self.force(u)
        return -f[self.free_mask]


class Operator(scipy.sparse.linalg.LinearOperator):
    model: Model

    def __init__(self, model: Model) -> None:
        self.model = model
        super().__init__(float, (self.model.n_free * 3, self.model.n_free * 3))

    def _matvec(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return self.model.Ax(x.reshape((self.model.n_free, 3))).flatten()


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    output_file: Annotated[
        pathlib.Path, typer.Option("-o", "--output", dir_okay=False, writable=True)
    ],
    poisson_ratio: Annotated[float, typer.Option()] = 0.0,
) -> None:
    ti.init(ti.cpu, default_fp=ti.float64)
    mesh: meshio.Mesh = mkit.io.load_meshio(input_file)
    model = Model(mesh)
    model.init_material(E=3000 * 1e-9, nu=poisson_ratio)
    model.volume()
    model.stiffness()
    A = Operator(model)
    u_free: npt.NDArray[np.floating]
    info: int
    u_free, info = scipy.sparse.linalg.minres(
        A, model.b().flatten(), show=True, rtol=1e-6
    )
    logger.info("GMRES info: {}", info)
    assert info == 0
    u_free = u_free.reshape((model.n_free, 3))
    disp: npt.NDArray[np.floating] = mesh.point_data["disp"]
    mesh.points[model.fixed_mask] += disp[model.fixed_mask]
    mesh.points[model.free_mask] += u_free
    mkit.io.save(output_file, mesh, point_data={"force": model.force_free(u_free)})


if __name__ == "__main__":
    mkit.cli.run(main)
