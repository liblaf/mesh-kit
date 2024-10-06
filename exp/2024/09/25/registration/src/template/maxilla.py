from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pydantic

import mkit
import mkit.typing.numpy as nt

if TYPE_CHECKING:
    import pyvista as pv


class Config(mkit.cli.BaseConfig):
    template: pydantic.FilePath = Path(
        "/home/liblaf/Documents/data/template/maxilla.vtp"
    )
    output: Path = Path("data/template/maxilla.vtp")


GROUP_NAMES_FREE: list[str] = ["nostrils", "upperTeeth"]
GROUP_NAMES_FITTING: list[str] = ["eyeSockets", "skull"]


def main(cfg: Config) -> None:
    maxilla: pv.PolyData = mkit.io.pyvista.load_poly_data(cfg.template)
    weight: nt.FN = np.zeros((maxilla.n_faces_strict,))
    weight[mkit.ops.select.select_by_group_names(GROUP_NAMES_FREE, mesh=maxilla)] = 0.1
    weight[mkit.ops.select.select_by_group_names(GROUP_NAMES_FITTING, mesh=maxilla)] = (
        1.0
    )
    maxilla.point_data.update(
        mkit.ops.attr.cell_data_to_point_data(maxilla, {"Weight": weight})
    )
    maxilla.rotate_x(180, inplace=True)
    mkit.io.save(maxilla, cfg.output)


mkit.cli.auto_run()(main)
