from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pydantic
import pydantic_settings

import mkit
import mkit.typing.numpy as nt

if TYPE_CHECKING:
    import pyvista as pv


class Config(mkit.cli.BaseConfig):
    model_config = pydantic_settings.SettingsConfigDict(
        cli_parse_args=True, yaml_file="params/non-rigid.yaml"
    )
    source: pydantic.FilePath = Path("data/rigid.vtp")
    output: Path = Path("data/non-rigid.vtp")
    steps: list[dict]


GROUPS_FITTING: list[str] = ["skull", "eyeSockets", "nostrils"]
GROUPS_FREE: list[str] = ["upperTeeth"]


def main(cfg: Config) -> None:
    source: pv.PolyData = mkit.io.pyvista.load_poly_data(cfg.source)
    target: pv.PolyData = mkit.ext.sculptor.get_template_maxilla()
    mkit.io.save(target, "data/target.vtp")
    mkit.io.save(source, "data/non-rigid-source.vtp")
    source_weight: nt.FN = np.ones((source.n_faces_strict,))
    source_weight[mkit.ops.select.select_by_group_names(GROUPS_FREE, mesh=source)] = 0
    source.point_data.update(
        mkit.ops.attr.cell_data_to_point_data(source, {"Weight": source_weight})
    )
    res: mkit.ops.registration.NonRigidRegistrationResult = (
        mkit.ops.registration.non_rigid_registration(
            mkit.ops.registration.non_rigid.Amberg(
                source,
                target,
                point_data={"weight": source.point_data["Weight"]},
                steps=cfg.steps,
            )
        )
    )
    result: pv.PolyData = source.copy()
    result.points = res.points
    mkit.io.save(result, cfg.output)


mkit.cli.auto_run()(main)
