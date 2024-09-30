from pathlib import Path

import numpy as np
import pydantic_settings
import pyvista as pv

import mkit
import mkit.typing.numpy as nt


class Config(mkit.cli.BaseConfig):
    source: pydantic_settings.CliPositionalArg[Path]


GROUP_NAMES_MATCH: list[str] = [
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
GROUP_NAMES_REMOVE: list[str] = [
    "MouthSocketBottom",
    "MouthSocketTop",
    "EyeSocketTop",
    "EyeSocketBottom",
]


def names_to_ids(mesh: pv.PolyData, names: list[str]) -> list[int]:
    return [i for i, name in enumerate(mesh.field_data["GroupNames"]) if name in names]


def select_by_group_names(mesh: pv.PolyData, names: list[str]) -> nt.BN:
    return np.isin(mesh.cell_data["GroupIds"], names_to_ids(mesh, names))


def main(cfg: Config) -> None:
    source: pv.PolyData = mkit.io.pyvista.load_poly_data(cfg.source)
    target: pv.PolyData = mkit.ext.sculptor.get_template_face()
    mkit.io.save(target, "data/target.vtp")
    source = source.remove_cells(select_by_group_names(source, GROUP_NAMES_REMOVE))
    source_weight: nt.FN = np.zeros((source.n_faces_strict,))
    source_weight[select_by_group_names(source, GROUP_NAMES_MATCH)] = 1
    source.cell_data["Weight"] = source_weight
    source = source.cell_data_to_point_data(pass_cell_data=True)
    mkit.io.save(source, "data/source.vtp")
    res: mkit.ops.registration.NonRigidRegistrationResult = (
        mkit.ops.registration.non_rigid_registration(source, target)
    )
    result: pv.PolyData = source.transform(res.result)
    mkit.io.save(result, "data/non-rigid.vtp")


mkit.cli.auto_run()(main)
