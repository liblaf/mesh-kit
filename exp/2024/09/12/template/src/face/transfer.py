from pathlib import Path

import pyvista as pv

import mkit


class Config(mkit.cli.BaseConfig):
    source: Path


@mkit.cli.auto_run()
def main(cfg: Config) -> None:
    source: pv.PolyData = mkit.io.pyvista.read_poly_data(cfg.source)
    ic(source.cell_data["GroupIds"])
    source = source.cell_data_to_point_data(progress_bar=True)
    target: pv.PolyData = mkit.ext.sculptor.get_template_face()
    target = mkit.ops.transfer.surface_to_surface(
        source, target, point_data={"GroupIds": source.point_data["GroupIds"]}
    )
    ic(target)
