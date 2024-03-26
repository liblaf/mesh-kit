import pathlib
from typing import Annotated, Any, no_type_check

import meshio
import numpy as np
import scipy.sparse.linalg
import taichi as ti
import typer
from mkit import cli as _cli
from mkit import io as _io
from mkit.fem import mtm as _mtm
from mkit.taichi.mesh import field as _field
from numpy import typing as npt


@no_type_check
@ti.func
def is_free(v: ti.template()) -> bool:
    return ti.math.isnan(v.disp).any()


@no_type_check
@ti.func
def is_fixed(v: ti.template()) -> bool:
    return not is_free(v)


@ti.data_oriented
class MTM(_mtm.MTM):
    free_mask: npt.NDArray

    def set_disp(self, disp: npt.NDArray) -> None:
        _field.place_safe(
            self.mesh.verts,
            {"Ax": ti.math.vec3, "b": ti.math.vec3, "disp": ti.math.vec3},
        )
        disp_ti: ti.MatrixField = self.mesh.verts.get_member_field("disp")
        disp_ti.from_numpy(disp)
        self.free_mask = np.isnan(disp).any(axis=1)

    def calc_b(self) -> None:
        b: ti.MatrixField = self.mesh.verts.get_member_field("b")
        b.fill(0)
        self._calc_b_kernel()

    def calc_Ax(self) -> None:
        Ax: ti.MatrixField = self.mesh.verts.get_member_field("Ax")
        Ax.fill(0)
        self._calc_Ax_kernel()

    def export_K11(self) -> scipy.sparse.csr_matrix:
        num_verts: int = len(self.mesh.verts)
        num_edges: int = len(self.mesh.edges)
        num_triplets: int = num_verts * 9 + 2 * num_edges * 9
        data_ti: ti.ScalarField = ti.field(dtype=float, shape=(num_triplets,))
        row_ti: ti.ScalarField = ti.field(dtype=int, shape=(num_triplets,))
        row_ti.fill(-1)
        col_ti: ti.ScalarField = ti.field(dtype=int, shape=(num_triplets,))
        col_ti.fill(-1)
        _field.place_safe(self.mesh.verts, {"free_id": int})
        free_id_np: npt.NDArray = np.full(len(self.mesh.verts), -1, int)
        free_id_np[self.free_mask] = np.arange(np.count_nonzero(self.free_mask))
        free_id_ti: ti.ScalarField = self.mesh.verts.get_member_field("free_id")
        free_id_ti.from_numpy(free_id_np)

        @no_type_check
        @ti.kernel
        def _kernel():
            for v in self.mesh.verts:
                if is_free(v):
                    for i, j in ti.ndrange(3, 3):
                        idx = v.id * 9 + i * 3 + j
                        data_ti[idx] = v.K[i, j]
                        row_ti[idx] = v.free_id * 3 + i
                        col_ti[idx] = v.free_id * 3 + j
            for e in self.mesh.edges:
                if is_free(e.verts[0]) and is_free(e.verts[1]):
                    for i, j in ti.ndrange(3, 3):
                        idx = num_verts * 9 + e.id * 18 + i * 3 + j
                        data_ti[idx] = e.K[i, j]
                        row_ti[idx] = e.verts[0].free_id * 3 + i
                        col_ti[idx] = e.verts[1].free_id * 3 + j
                        data_ti[idx + 9] = e.K[i, j]
                        row_ti[idx + 9] = e.verts[1].free_id * 3 + j
                        col_ti[idx + 9] = e.verts[0].free_id * 3 + i

        _kernel()
        data_np = data_ti.to_numpy()
        row_np = row_ti.to_numpy()
        col_np = col_ti.to_numpy()
        valid = row_np != -1
        mat = scipy.sparse.csr_matrix((data_np[valid], (row_np[valid], col_np[valid])))
        return mat

    @property
    def fixed_mask(self) -> npt.NDArray:
        return ~self.free_mask

    @no_type_check
    @ti.kernel
    def _calc_b_kernel(self):
        """b = - K10 @ x0"""
        for v in self.mesh.verts:
            if is_free(v):
                for e_idx in range(v.edges.size):
                    e = v.edges[e_idx]
                    if e.verts[0].id == v.id and is_fixed(e.verts[1]):
                        v.b -= e.K @ e.verts[1].disp
                    elif e.verts[1].id == v.id and is_fixed(e.verts[0]):
                        v.b -= e.K.transpose() @ e.verts[0].disp

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
        x_np: npt.NDArray = np.zeros((len(self.mtm.mesh.verts), 3))
        x_np[self.mtm.free_mask] = x.reshape((-1, 3))
        x_ti: ti.MatrixField = self.mtm.mesh.verts.get_member_field("x")
        x_ti.from_numpy(x_np)
        self.mtm.calc_Ax()
        Ax_ti: ti.MatrixField = self.mtm.mesh.verts.get_member_field("Ax")
        Ax_np: npt.NDArray = Ax_ti.to_numpy()
        return Ax_np[self.mtm.free_mask].flatten()

    def _rmatvec(self, x: npt.NDArray) -> npt.NDArray:
        return self._matvec(x)


def main(
    tet_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    output_dir: Annotated[
        pathlib.Path, typer.Option(exists=True, file_okay=False, writable=True)
    ],
    young_modulus: Annotated[float, typer.Option(min=0)] = 3000,
    poisson_ratio: Annotated[float, typer.Option(min=0, max=0.5)] = 0.2,
    step: Annotated[int, typer.Option(min=0)] = 10,
) -> None:
    ti.init(default_fp=ti.f64)
    mesh_io: meshio.Mesh = meshio.read(tet_file)
    mesh_ti: ti.MeshInstance = _io.to_taichi(
        mesh_io, relations=["CE", "CV", "EV", "VE"]
    )
    disp: npt.NDArray = np.asarray(mesh_io.point_data["disp"])
    mtm: MTM = MTM(mesh_ti)
    mtm.init(E=young_modulus, nu=poisson_ratio)
    mtm.calc_stiffness()
    mtm.set_disp(disp)
    K11: scipy.sparse.csr_matrix = mtm.export_K11()

    steps: npt.NDArray = np.linspace(0, disp, step)
    for i, disp in enumerate(steps):
        mtm.set_disp(disp)
        mtm.calc_b()
        b_ti: ti.MatrixField = mtm.mesh.verts.get_member_field("b")
        b_np: npt.NDArray = b_ti.to_numpy()
        x_free_flatten: npt.NDArray = scipy.sparse.linalg.spsolve(
            K11, b_np[mtm.free_mask].flatten()
        )
        x_free: npt.NDArray = x_free_flatten.reshape((-1, 3))
        result: meshio.Mesh = mesh_io.copy()
        result.points[mtm.fixed_mask] += disp[mtm.fixed_mask]
        result.points[mtm.free_mask] += x_free

        x_ti: ti.MatrixField = mtm.mesh.verts.get_member_field("x")
        x_ti.from_numpy(result.points - mesh_io.points)
        mtm.calc_force()
        f_ti: ti.MatrixField = mtm.mesh.verts.get_member_field("f")
        f_np: npt.NDArray = f_ti.to_numpy()
        result.point_data["f"] = f_np
        result.write(output_dir / f"{i:02d}.vtu")


if __name__ == "__main__":
    _cli.run(main)
