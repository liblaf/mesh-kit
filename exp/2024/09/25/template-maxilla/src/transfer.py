from pathlib import Path
from typing import TYPE_CHECKING

import pydantic

import mkit

if TYPE_CHECKING:
    import pyvista as pv


class Config(mkit.cli.BaseConfig):
    source: pydantic.FilePath = Path("data/non-rigid.vtp")
    output: Path = Path("data/transfer.vtp")


def main(cfg: Config) -> None:
    source: pv.PolyData = mkit.io.pyvista.load_poly_data(cfg.source)
    target: pv.PolyData = mkit.ext.sculptor.get_template_maxilla()
    target.triangulate(inplace=True)
    target.cell_data.update(
        mkit.ops.transfer.surface_to_surface(
            source,
            target,
            {"GroupIds": source.cell_data["GroupIds"]},
            method=mkit.ops.transfer.C2CAuto(),
        )
    )
    target.field_data["GroupNames"] = source.field_data["GroupNames"]
    mkit.io.save(target, cfg.output)


mkit.cli.auto_run()(main)
