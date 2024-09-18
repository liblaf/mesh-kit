import pyvista as pv

import mkit


class Config(mkit.cli.BaseConfig):
    pass


@mkit.cli.auto_run()
def main(cfg: Config) -> None:
    source: pv.PolyData = mkit.ext.sculptor.get_template_skull()
    target: pv.PolyData = mkit.ext.sculptor.get_template_face()
