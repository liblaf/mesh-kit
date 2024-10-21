from pathlib import Path
from typing import TYPE_CHECKING

import mkit.io as mi
import mkit.utils as mu

if TYPE_CHECKING:
    import pyvista as pv


class Config(mu.cli.BaseConfig):
    path: Path


def main(cfg: Config) -> None:
    mesh: pv.PolyData = mi.pyvista.load_poly_data(cfg.path)
    ic(mesh.field_data["GroupNames"])


mu.cli.auto_run()(main)
