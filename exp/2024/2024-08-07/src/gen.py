import dataclasses

import numpy as np
import pyvista as pv
import rich
import rich.traceback
import taichi as ti
from icecream import ic
from omegaconf import OmegaConf

import mkit.ext
import mkit.logging
import mkit.point
import mkit.sparse

ti.init()
mkit.logging.init()
rich.traceback.install(show_locals=True)


@dataclasses.dataclass(kw_only=True)
class Config:
    E: float = 1e3
    nu: float = 0.46


def main() -> None:
    cfg: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    ic(cfg)
    E: float = cfg.E
    nu: float = cfg.nu
    mu: float = E / (2 * (1 + nu))
    lambda_: float = E * nu / ((1 + nu) * (1 - 2 * nu))
    ic(mu, lambda_)

    inner: pv.PolyData = pv.Icosphere(radius=0.1)
    inner.point_data["pin_mask"] = np.ones((inner.n_points,), bool)
    pin_disp = np.zeros((inner.n_points, 3))
    # pin_disp[:, 1] = 1.0 * inner.points[:, 1]
    inner.point_data["pin_disp"] = np.zeros((inner.n_points, 3))
    inner.point_data["pin_disp"] = pin_disp
    inner.point_data["normal"] = inner.point_normals

    outer: pv.PolyData = pv.Icosphere(radius=0.2)
    outer.point_data["pin_mask"] = np.zeros((outer.n_points,), bool)
    outer.point_data["pin_disp"] = np.zeros((outer.n_points, 3))
    outer.point_data["normal"] = outer.point_normals

    mesh: pv.UnstructuredGrid = mkit.ext.tetwild_wrapped(outer, inner)
    ic(mesh)
    mesh.cell_data["mu"] = np.full((mesh.n_cells,), mu)
    mesh.cell_data["lambda"] = np.full((mesh.n_cells,), lambda_)
    mesh.cell_data["density"] = np.full((mesh.n_cells,), 1e3)
    mesh.cell_data["gravity"] = np.tile(np.asarray([0, -9.8, 0]), (mesh.n_cells, 1))

    mesh.save("data/input.vtu")


if __name__ == "__main__":
    main()
