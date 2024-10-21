from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh.transformations as tf

import mkit.ext as me
import mkit.io as mi
import mkit.ops as mo
import mkit.utils as mu

if TYPE_CHECKING:
    import pyvista as pv


class Config(mu.cli.BaseConfig):
    source: Path = Path(
        "~/.local/opt/Wrap/Gallery/TexturingXYZ/XYZ_ReadyToSculpt_eyesOpen_PolyGroups_GEO.obj"
    ).expanduser()


def main(cfg: Config) -> None:
    source: pv.PolyData = mi.pyvista.load_poly_data(cfg.source)
    target: pv.PolyData = me.sculptor.get_template_face()
    ic(source.field_data["GroupNames"])
    rigid = mo.RigidICP(
        source, target, init_transform=tf.rotation_matrix(np.pi / 2, [1, 0, 0])
    )
    result: mo.RigidRegistrationResult = rigid.register()
    mi.save("data/source.obj", source)
    mi.save("data/target.obj", target)
    mi.save("data/rigid.obj", mo.transform(source, result.transformation))


mu.cli.auto_run()(main)
