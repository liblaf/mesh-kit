import pyvista as pv

import mkit
from mkit.ops.registration.non_rigid import non_rigid_registration


class Config(mkit.cli.BaseConfig):
    pass


@mkit.cli.auto_run()
def main(cfg: Config) -> None:
    source: pv.PolyData = pv.Icosphere(radius=1.0)
    target: pv.PolyData = pv.Icosphere(radius=1.1)
    result: pv.PolyData = non_rigid_registration(source, target)
    source.save("source.vtp")
    target.save("target.vtp")
    result.save("result.vtp")
