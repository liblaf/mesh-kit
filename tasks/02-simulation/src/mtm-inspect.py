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
from icecream import ic
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


def main(
    input_file: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    poisson_ratio: Annotated[float, typer.Option()] = 0.01,
) -> None:
    ti.init(ti.cpu, default_fp=ti.float64, debug=True)
    mesh: meshio.Mesh = mkit.io.load_meshio(input_file)
    model = Model(mesh)
    model.init_material(E=3000 * 1e-9, nu=poisson_ratio)
    model.volume()
    model.stiffness()
    stiffness: npt.NDArray[np.floating] = model.stiffness_sparse_matrix().todense()
    free_mask: npt.NDArray[np.bool_] = np.repeat(model.free_mask, 3)
    stiffness = stiffness[free_mask][:, free_mask]
    ic(np.linalg.cond(stiffness))
    print(poisson_ratio, np.linalg.cond(stiffness), sep=",")


if __name__ == "__main__":
    mkit.cli.run(main)
