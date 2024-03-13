import pathlib
from typing import Annotated, no_type_check

import meshtaichi_patcher
import numpy as np
import taichi as ti
import trimesh
import typer
from icecream import ic
from mesh_kit import cli, points
from mesh_kit.io import node
from mesh_kit.linalg import cg
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
        fixed_np: npt.NDArray = np.full((len(self.mesh.verts), 3), np.nan)
        fixed_idx: npt.NDArray = points.position2idx(
            self.mesh.get_position_as_numpy(), pre_skull.vertices
        )
        self.fixed_mask = np.full(len(self.mesh.verts), False)
        self.fixed_mask[fixed_idx] = True
        fixed_np[self.fixed_mask] = post_skull.vertices
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
    K: npt.NDArray = mtm.export_stiffness()
    ic(np.linalg.cond(K))
    K11_mask: npt.NDArray = np.repeat(~mtm.fixed_mask, 3)
    K11: npt.NDArray = K[np.ix_(K11_mask, K11_mask)]
    ic(K11.shape)
    ic(np.linalg.cond(K11))
    mtm.calc_b()
    b: ti.MatrixField = mtm.mesh.verts.get_member_field("b")
    x: ti.MatrixField = mtm.mesh.verts.get_member_field("x")
    x.fill(0)
    converge: bool = cg.cg(ti.linalg.LinearOperator(mtm.calc_Ax), b, x)
    assert converge
    node.save(
        pathlib.Path("data") / "post.1.node",
        mtm.mesh.get_position_as_numpy() + x.to_numpy(),
        comment=False,
    )


if __name__ == "__main__":
    cli.run(main)
