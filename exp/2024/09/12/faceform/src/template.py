from typing import TYPE_CHECKING

import mkit

if TYPE_CHECKING:
    import pyvista as pv


class Config(mkit.cli.BaseConfig):
    pass


@mkit.cli.auto_run()
def main(cfg: Config) -> None:
    face: pv.PolyData = mkit.ext.sculptor.get_template_face()
    face.save("data/template/face.obj")
    maxilla: pv.PolyData = mkit.ext.sculptor.get_template_maxilla()
    maxilla.save("data/template/maxilla.obj")
    mandible: pv.PolyData = mkit.ext.sculptor.get_template_mandible()
    mandible.save("data/template/mandible.obj")
