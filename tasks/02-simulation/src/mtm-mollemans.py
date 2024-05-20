import functools
import pathlib
from typing import Annotated, no_type_check

import meshio
import mkit.cli
import mkit.io
import mkit.physics.mtm
import mkit.taichi.mesh.field
import numpy as np
import taichi as ti
import typer
from numpy import typing as npt


@ti.data_oriented
class Model(mkit.physics.mtm.MTM):
    def __init__(self, mesh: meshio.Mesh) -> None:
        super().__init__(mesh, ["CE", "CV", "EV", "VE"])
        mkit.taichi.mesh.field.place(self.mesh.verts, {"disp": ti.math.vec3})
        disp: ti.MatrixField = self.mesh.verts.get_member_field("disp")
        disp.from_numpy(mesh.point_data["disp"])

    @functools.cached_property
    def disp(self) -> npt.NDArray[np.floating]:
        disp_ti: ti.MatrixField = self.mesh.verts.get_member_field("disp")
        disp_np: npt.NDArray[np.floating] = disp_ti.to_numpy()
        return disp_np

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
        u: npt.NDArray[np.floating] = np.zeros((self.n_verts, 3))
        u[self.fixed_mask] = self.disp[self.fixed_mask]
        u[self.free_mask] = u_free
        f: npt.NDArray[np.floating] = self.force(u)
        return f

    def step(self, u_free: npt.ArrayLike) -> npt.NDArray[np.floating]:
        mkit.taichi.mesh.field.place(
            self.mesh.verts, {"u": ti.math.vec3, "u_new": ti.math.vec3}
        )
        u_np: npt.NDArray[np.floating] = np.zeros((self.n_verts, 3))
        u_np[self.fixed_mask] = self.disp[self.fixed_mask]
        u_np[self.free_mask] = u_free
        u_ti: ti.MatrixField = self.mesh.verts.get_member_field("u")
        u_ti.from_numpy(u_np)
        self._step()
        u_new_ti: ti.MatrixField = self.mesh.verts.get_member_field("u_new")
        u_new_np: npt.NDArray[np.floating] = u_new_ti.to_numpy()
        return u_new_np[self.free_mask]

    @no_type_check
    @ti.kernel
    def _step(self):
        for v in self.mesh.verts:
            f = ti.Vector.zero(float, 3)
            for e_idx in range(v.edges.size):
                e = v.edges[e_idx]
                if e.verts[0].id == v.id:
                    f += e.K @ e.verts[1].u
                elif e.verts[1].id == v.id:
                    f += e.K.transpose() @ e.verts[0].u
            v.u_new = -v.K.inverse() @ f


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
    u_free: npt.NDArray[np.floating] = np.zeros((model.n_free, 3))
    for i in range(1000):
        u_free = model.step(u_free)
    disp: npt.NDArray[np.floating] = mesh.point_data["disp"]
    mesh.points[model.fixed_mask] += disp[model.fixed_mask]
    mesh.points[model.free_mask] += u_free
    mkit.io.save(output_file, mesh, point_data={"force": model.force_free(u_free)})


if __name__ == "__main__":
    mkit.cli.run(main)
