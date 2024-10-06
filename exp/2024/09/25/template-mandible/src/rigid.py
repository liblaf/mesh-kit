from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pydantic_settings
import torch
import trimesh.transformations as tt

import mkit
import mkit.ops.registration.rigid as reg
import mkit.typing.numpy as nt

if TYPE_CHECKING:
    import pyvista as pv


class Config(mkit.cli.BaseConfig):
    source: pydantic_settings.CliPositionalArg[Path]
    output: Path = Path("data/rigid.vtp")


GROUPS_TO_KEEP: list[str] = ["jaw"]
GROUPS_TO_REMOVE: list[str] = ["lowerTeeth"]


def main(cfg: Config) -> None:
    torch.set_default_device("cuda")
    source: pv.PolyData = mkit.io.pyvista.load_poly_data(cfg.source)
    target: pv.PolyData = mkit.ext.sculptor.get_template_mandible()
    mkit.io.save(target, "data/target.vtp")
    mkit.io.save(source, "data/source.vtp")
    source_weight: nt.FN = np.ones((source.n_faces_strict,))
    source_weight[
        mkit.ops.select.select_by_group_names(GROUPS_TO_REMOVE, mesh=source)
    ] = 0
    source.point_data.update(
        mkit.ops.attr.cell_data_to_point_data(source, {"Weight": source_weight})
    )
    res: reg.RigidRegistrationResult = reg.rigid_registration(
        source,
        target,
        estimate_init=True,
        init=tt.rotation_matrix(np.pi / 2, [1, 0, 0]),
        source_weight=source.point_data["Weight"],
    )
    ic(res.cost)
    result: pv.PolyData = source.transform(res.transform)
    mkit.io.save(result, cfg.output)


mkit.cli.auto_run()(main)
