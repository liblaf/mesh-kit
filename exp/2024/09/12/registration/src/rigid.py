from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import mkit

if TYPE_CHECKING:
    import pyvista as pv

    from mkit.ops.registration.rigid._result import RigidRegistrationResult


class Config(mkit.cli.BaseConfig):
    template: Path


@mkit.cli.auto_run()
def main(cfg: Config) -> None:
    source: pv.PolyData = mkit.io.pyvista.read_poly_data(cfg.template)
    target: pv.PolyData = mkit.ext.sculptor.get_template_skull()
    source.save("data/source.vtp")
    target.save("data/target.vtp")
    result: RigidRegistrationResult = mkit.ops.registration.rigid_registration(
        source, target
    )
    source.transform(result.transform, inplace=True)
    ic(np.unique(source.cell_data["GroupIds"]))
    ic(source.field_data["GroupNames"])
    source.save("data/result.vtp")
