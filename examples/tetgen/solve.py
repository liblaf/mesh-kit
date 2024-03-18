import pathlib
from typing import Annotated, no_type_check

import meshtaichi_patcher
import numpy as np
import scipy.sparse.linalg
import taichi as ti
import trimesh
import trimesh.registration
import typer
from icecream import ic
from mesh_kit import cli, points
from mesh_kit.io import node
from mesh_kit.physics import mtm as _mtm
from mesh_kit.typing import check_type as _t
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

    def init_fixed(
        self,
        pre_skull_file: pathlib.Path,
        post_skull_file: pathlib.Path,
        *,
        young_modulus: float = 3000,
        poisson_ratio: float = 0.47,
    ) -> None:
        super().init(young_modulus, poisson_ratio)
        self.mesh.verts.place(
            {"Ax": ti.math.vec3, "b": ti.math.vec3, "fixed": ti.math.vec3}
        )
        fixed: ti.MatrixField = self.mesh.verts.get_member_field("fixed")
        pre_skull: trimesh.Trimesh = _t(trimesh.Trimesh, trimesh.load(pre_skull_file))
        post_skull: trimesh.Trimesh = _t(trimesh.Trimesh, trimesh.load(post_skull_file))
        # post_skull.apply_translation(pre_skull.centroid - post_skull.centroid)
        fixed_np: npt.NDArray = np.full((len(self.mesh.verts), 3), np.nan)
        fixed_idx: npt.NDArray = points.position2idx(
            self.mesh.get_position_as_numpy(), pre_skull.vertices
        )
        self.fixed_mask = np.full(len(self.mesh.verts), False)
        self.fixed_mask[fixed_idx] = True
        fixed_np[fixed_idx] = post_skull.vertices
        fixed.from_numpy(fixed_np)

    def calc_b(self) -> None:
        b: ti.MatrixField = self.mesh.verts.get_member_field("b")
        b.fill(0)
        self._calc_b_kernel()

    def calc_Ax(self, x: ti.MatrixField, Ax: ti.MatrixField) -> None:
        x_mesh: ti.MatrixField = self.mesh.verts.get_member_field("x")
        x_mesh.copy_from(x)
        self._calc_Ax_kernel()
        Ax_mesh: ti.MatrixField = self.mesh.verts.get_member_field("Ax")
        Ax.copy_from(Ax_mesh)

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

    def __init__(self, dtype, mtm: MTM) -> None:
        num_free: int = np.count_nonzero(mtm.free_mask)
        super().__init__(dtype, shape=(num_free * 3, num_free * 3))
        self.mtm = mtm

    def _matvec(self, x: npt.NDArray) -> npt.NDArray:
        x_field: ti.MatrixField = self.mtm.mesh.verts.get_member_field("x")
        Ax: ti.MatrixField = self.mtm.mesh.verts.get_member_field("Ax")
        x_np: npt.NDArray = np.zeros((len(self.mtm.mesh.verts), 3))
        x_np[self.mtm.free_mask] = x.reshape((-1, 3))
        x_field.from_numpy(x_np)
        self.mtm.calc_Ax(x_field, Ax)
        return Ax.to_numpy()[self.mtm.free_mask].flatten()

    def _rmatvec(self, x: npt.NDArray) -> npt.NDArray:
        return self._matvec(x)


def main(
    tet_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    pre_skull_file: Annotated[
        pathlib.Path, typer.Option("--pre", exists=True, dir_okay=False)
    ],
    post_skull_file: Annotated[
        pathlib.Path, typer.Option("--post", exists=True, dir_okay=False)
    ],
    young_modulus: Annotated[float, typer.Option(min=0)] = 3000,
    poisson_ratio: Annotated[float, typer.Option(min=0, max=0.5)] = 0.47,
) -> None:
    ti.init(default_fp=ti.f64)
    mesh: ti.MeshInstance = meshtaichi_patcher.load_mesh(
        str(tet_file), relations=["EV", "CV", "CE", "VE"]
    )
    mtm: MTM = MTM(mesh)
    mtm.init_fixed(
        pre_skull_file=pre_skull_file,
        post_skull_file=post_skull_file,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
    )
    mtm.calc_stiffness()
    mtm.calc_b()
    b: ti.MatrixField = mtm.mesh.verts.get_member_field("b")
    x: ti.MatrixField = mtm.mesh.verts.get_member_field("x")
    x.fill(0)
    x_free_np: npt.NDArray
    pos_np: npt.NDArray = mtm.mesh.get_position_as_numpy()
    fixed: ti.MatrixField = mtm.mesh.verts.get_member_field("fixed")
    fixed_np: npt.NDArray = fixed.to_numpy()
    matrix, _, cost = trimesh.registration.procrustes(
        pos_np[mtm.fixed_mask], fixed_np[mtm.fixed_mask]
    )
    ic(cost)
    transformed = trimesh.transform_points(pos_np, matrix)
    x0: npt.NDArray = (transformed[mtm.free_mask] - pos_np[mtm.free_mask]).flatten()
    print(x0)
    x_free_np, info = scipy.sparse.linalg.minres(
        Operator(float, mtm=mtm),
        b.to_numpy()[mtm.free_mask].flatten(),
        x0=x0,
        show=True,
    )
    ic(np.abs(x_free_np).max())
    ic(x_free_np, info)
    pos_np: npt.NDArray = mtm.mesh.get_position_as_numpy()
    pos_np[mtm.free_mask] += x_free_np.reshape((-1, 3))
    pos_np[mtm.fixed_mask] = mtm.mesh.verts.get_member_field("fixed").to_numpy()[
        mtm.fixed_mask
    ]
    node.save(pathlib.Path("data") / "post.1.node", pos_np, comment=False)


if __name__ == "__main__":
    cli.run(main)
