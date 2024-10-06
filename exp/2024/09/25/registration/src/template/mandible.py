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
        "/home/liblaf/Documents/data/template/mandible.vtp"
    )
    output: Path = Path("data/template/mandible.vtp")


GROUP_NAMES_FREE: list[str] = ["lowerTeeth"]
GROUP_NAMES_FITTING: list[str] = ["jaw"]


def main(cfg: Config) -> None:
    mandible: pv.PolyData = mkit.io.pyvista.load_poly_data(cfg.template)
    weight: nt.FN = np.zeros((mandible.n_faces_strict,))
    weight[mkit.ops.select.select_by_group_names(GROUP_NAMES_FREE, mesh=mandible)] = 0.1
    weight[
        mkit.ops.select.select_by_group_names(GROUP_NAMES_FITTING, mesh=mandible)
    ] = 1.0
    mandible.point_data.update(
        mkit.ops.attr.cell_data_to_point_data(mandible, {"Weight": weight})
    )
    mandible.rotate_x(180, inplace=True)
    mkit.io.save(mandible, cfg.output)


mkit.cli.auto_run()(main)
