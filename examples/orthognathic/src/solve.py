import pathlib
from typing import Annotated, Any, no_type_check

import meshio
import numpy as np
import scipy.sparse.linalg
import taichi as ti
import typer
from mkit import cli as _cli
from mkit.fem import mtm as _mtm
from mkit.io.taichi import mesh as io_ti_mesh
from mkit.taichi.mesh import field as _field
from numpy import typing as npt


@no_type_check
@ti.func
def is_free(v: ti.template()) -> bool:
    return ti.math.isnan(v.fixed).any()


@no_type_check
@ti.func
def is_fixed(v: ti.template()) -> bool:
    return not is_free(v)


@ti.data_oriented
class MTM(_mtm.MTM):
    fixed_mask: npt.NDArray

    def init_fixed(self, fixed: npt.NDArray, *, E: float, nu: float) -> None:
        super().init(E=E, nu=nu)
        _field.place_safe(
            self.mesh.verts,
            {"Ax": ti.math.vec3, "b": ti.math.vec3, "fixed": ti.math.vec3},
        )
        fixed_ti: ti.MatrixField = self.mesh.verts.get_member_field("fixed")
        fixed_ti.from_numpy(fixed)
        self.fixed_mask = ~np.isnan(fixed).any(axis=1)

    def calc_b(self) -> None:
        b: ti.MatrixField = self.mesh.verts.get_member_field("b")
        b.fill(0)
        self._calc_b_kernel()

    def calc_Ax(self, x: ti.MatrixField, Ax: ti.MatrixField) -> None:
        x_ti: ti.MatrixField = self.mesh.verts.get_member_field("x")
        x_ti.copy_from(x)
        self._calc_Ax_kernel()
        Ax_ti: ti.MatrixField = self.mesh.verts.get_member_field("Ax")
        Ax.copy_from(Ax_ti)

    @property
    def free_mask(self) -> npt.NDArray:
        return ~self.fixed_mask

    @no_type_check
    @ti.kernel
    def _calc_b_kernel(self):
        """b = - K10 @ x0"""
        for v in self.mesh.verts:
            if is_free(v):
                for e_idx in range(v.edges.size):
                    e = v.edges[e_idx]
                    if e.verts[0].id == v.id and is_fixed(e.verts[1]):
                        v.b -= e.K @ (e.verts[1].fixed - e.verts[1].pos)
                    elif e.verts[1].id == v.id and is_fixed(e.verts[0]):
                        v.b -= e.K.transpose() @ (e.verts[0].fixed - e.verts[0].pos)

    @no_type_check
    @ti.kernel
    def _calc_Ax_kernel(self):
        """Ax = K11 @ x1"""
        for v in self.mesh.verts:
            if is_free(v):
                v.Ax += v.K @ v.x
                for e_idx in range(v.edges.size):
                    e = v.edges[e_idx]
                    if e.verts[0].id == v.id and is_free(e.verts[1]):
                        v.Ax += e.K @ e.verts[1].x
                    elif e.verts[1].id == v.id and is_free(e.verts[0]):
                        v.Ax += e.K.transpose() @ e.verts[0].x


class Operator(scipy.sparse.linalg.LinearOperator):
    mtm: MTM

    def __init__(self, dtype: Any, mtm: MTM) -> None:
        num_free: int = np.count_nonzero(mtm.free_mask)
        super().__init__(dtype, shape=(num_free * 3, num_free * 3))
        self.mtm = mtm

    def _matvec(self, x: npt.NDArray) -> npt.NDArray:
        x_ti: ti.MatrixField = self.mtm.mesh.verts.get_member_field("x")
        Ax_ti: ti.MatrixField = self.mtm.mesh.verts.get_member_field("Ax")
        x_np: npt.NDArray = np.zeros((len(self.mtm.mesh.verts), 3))
        x_np[self.mtm.free_mask] = x.reshape((-1, 3))
        x_ti.from_numpy(x_np)
        self.mtm.calc_Ax(x_ti, Ax_ti)
        Ax_np: npt.NDArray = Ax_ti.to_numpy()
        return Ax_np[self.mtm.free_mask].flatten()

    def _rmatvec(self, x: npt.NDArray) -> npt.NDArray:
        return self._matvec(x)


def main(
    tet_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    output_dir: Annotated[
        pathlib.Path,
        typer.Option("-o", "--output", exists=True, file_okay=False, writable=True),
    ],
    young_modulus: Annotated[float, typer.Option(min=0)] = 3000,
    poisson_ratio: Annotated[float, typer.Option(min=0, max=0.5)] = 0.47,
) -> None:
    ti.init(default_fp=ti.f64)
    mesh_io: meshio.Mesh = meshio.read(tet_file)
    mesh: ti.MeshInstance = io_ti_mesh.from_meshio(
        mesh_io, relations=["EV", "CV", "CE", "VE"]
    )
    fixed_np: npt.NDArray = np.asarray(mesh_io.point_data["fixed"])
    mtm: MTM = MTM(mesh)
    mtm.init_fixed(fixed_np, E=young_modulus, nu=poisson_ratio)
    mtm.calc_stiffness()
    mtm.calc_b()
    b_ti: ti.MatrixField = mtm.mesh.verts.get_member_field("b")
    x0: npt.NDArray = np.zeros((np.count_nonzero(mtm.free_mask) * 3,))
    x_free_np: npt.NDArray
    info: int
    x_free_np, info = scipy.sparse.linalg.minres(
        Operator(float, mtm=mtm),
        b_ti.to_numpy()[mtm.free_mask].flatten(),
        x0=x0,
        show=True,
    )
    assert info == 0
    pos_np: npt.NDArray = mesh_io.points.copy()
    pos_np[mtm.fixed_mask] = fixed_np[mtm.fixed_mask]
    pos_np[mtm.free_mask] += x_free_np.reshape((-1, 3))
    meshio.write(
        output_dir / "0.vtu",
        meshio.Mesh(points=pos_np, cells={"tetra": mesh_io.get_cells_type("tetra")}),
    )


if __name__ == "__main__":
    _cli.run(main)
