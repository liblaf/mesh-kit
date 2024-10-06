from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pydantic

import mkit
import mkit.typing.numpy as nt

if TYPE_CHECKING:
    import pyvista as pv


class Config(mkit.cli.BaseConfig):
    template: pydantic.FilePath = Path("/home/liblaf/Documents/data/template/face.vtp")
    output: Path = Path("data/template/face.vtp")


GROUP_NAMES_REMOVE: list[str] = [
    "EyeSocketBottom",
    "EyeSocketTop",
    "MouthSocketBottom",
    "MouthSocketTop",
]
GROUP_NAMES_FREE: list[str] = [
    "EarSocket",
    "LipInnerBottom",
    "LipInnerTop",
    "Nostril",
]
GROUP_NAMES_WEAK: list[str] = [
    "Ear",
    "EarNeckBack",
]
GROUP_NAMES_PARTIAL: list[str] = [
    "HeadBack",
    "NeckBack",
    "NeckFront",
]
GROUP_NAMES_FITTING: list[str] = [
    "Caruncle",
    "Chin",
    "EyelidBottom",
    "EyelidInnerBottom",
    "EyelidInnerTop",
    "EyelidOuterBottom",
    "EyelidOuterTop",
    "EyelidTop",
    "Face",
    "LipBottom",
    "LipOuterBottom",
    "LipOuterTop",
    "LipTop",
]


def main(cfg: Config) -> None:
    face: pv.PolyData = mkit.io.pyvista.load_poly_data(cfg.template)
    face.remove_cells(
        mkit.ops.select.select_by_group_names(GROUP_NAMES_REMOVE, mesh=face),
        inplace=True,
    )
    weight: nt.FN = np.zeros((face.n_faces_strict,))
    weight[mkit.ops.select.select_by_group_names(GROUP_NAMES_FREE, mesh=face)] = 0.0
    weight[mkit.ops.select.select_by_group_names(GROUP_NAMES_WEAK, mesh=face)] = 0.1
    weight[mkit.ops.select.select_by_group_names(GROUP_NAMES_PARTIAL, mesh=face)] = 0.4
    weight[mkit.ops.select.select_by_group_names(GROUP_NAMES_FITTING, mesh=face)] = 1.0
    face.point_data.update(
        mkit.ops.attr.cell_data_to_point_data(face, {"Weight": weight})
    )
    face.rotate_x(180, inplace=True)
    mkit.io.save(face, cfg.output)


mkit.cli.auto_run()(main)
