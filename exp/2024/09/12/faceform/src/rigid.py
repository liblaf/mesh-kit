from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh as tm

import mkit

if TYPE_CHECKING:
    import pyvista as pv

    from mkit.ops.registration import GlobalRegistrationResult, RigidRegistrationResult


class Config(mkit.cli.BaseConfig):
    target: Path
    output: Path


@mkit.cli.auto_run()
def main(cfg: Config) -> None:
    source: pv.PolyData = mkit.ext.sculptor.get_template_skull()
    target: pv.PolyData = mkit.io.pyvista.read_poly_data(cfg.target)
    source.save("data/source.vtp")
    target.save("data/target.vtp")
    result: GlobalRegistrationResult = mkit.ops.registration.global_registration(
        source, target, init=tm.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    )
    source.transform(result.transform, inplace=True)
    source.save("data/global.obj")
    result: RigidRegistrationResult = mkit.ops.registration.rigid_registration(
        source, target
    )
    source.transform(result.transform, inplace=True)
    source.save(cfg.output)
