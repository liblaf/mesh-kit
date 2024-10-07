from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pydantic
import scipy.spatial

import mkit
import mkit.typing.numpy as nt

if TYPE_CHECKING:
    import pyvista as pv


class Config(mkit.cli.BaseConfig):
    template: pydantic.FilePath = Path(
        "/home/liblaf/Documents/data/template/maxilla.vtp"
    )
    mandible: pydantic.FilePath = Path(
        "/home/liblaf/Documents/data/template/mandible.vtp"
    )
    output: Path = Path("data/template/maxilla.vtp")


GROUP_NAMES_FREE: list[str] = ["nostrils", "upperTeeth"]
GROUP_NAMES_FITTING: list[str] = ["eyeSockets", "skull"]


def main(cfg: Config) -> None:
    maxilla: pv.PolyData = mkit.io.pyvista.load_poly_data(cfg.template)
    mandible: pv.PolyData = mkit.io.pyvista.load_poly_data(cfg.mandible)
    weight: nt.FN = np.zeros((maxilla.n_faces_strict,))
    weight[mkit.ops.select.select_by_group_names(GROUP_NAMES_FREE, mesh=maxilla)] = 0.1
    weight[mkit.ops.select.select_by_group_names(GROUP_NAMES_FITTING, mesh=maxilla)] = (
        1.0
    )
    maxilla.point_data.update(
        mkit.ops.attr.cell_data_to_point_data(maxilla, {"Weight": weight})
    )
    kdtree: scipy.spatial.KDTree = scipy.spatial.KDTree(mandible.points)
    dist: nt.FN
    dist, _idx = kdtree.query(maxilla.points)
    maxilla.point_data["Weight"][dist < 0.02 * maxilla.length] = 0.0
    maxilla.rotate_x(180, inplace=True)
    mkit.io.save(maxilla, cfg.output)


mkit.cli.auto_run()(main)
