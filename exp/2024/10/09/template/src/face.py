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


GROUP_NAMES_MATCH: list[str] = [
    "Caruncle",
    "Chin",
    "EyelidBottom",
    "EyelidInnerBottom",
    "EyelidInnerTop",
    "EyelidOuterBottom",
    "EyelidOuterTop",
    "EyelidTop",
    "Face",
    "LipBottom",
    "LipOuterBottom",
    "LipOuterTop",
    "LipTop",
]
GROUP_NAMES_REMOVE: list[str] = [
    "MouthSocketBottom",
    "MouthSocketTop",
    "EyeSocketTop",
    "EyeSocketBottom",
]


def main(cfg: Config) -> None:
    source: pv.PolyData = mi.pyvista.load_poly_data(cfg.source)
    target: pv.PolyData = me.sculptor.get_template_face()
    source = mo.select_by_group_names(source, GROUP_NAMES_REMOVE, invert=True)
    rigid = mo.RigidICP(
        source,
        target,
        source_weights=mo.mask_by_group_names(source, GROUP_NAMES_MATCH),
        init_transform=tf.rotation_matrix(np.pi / 2, [1, 0, 0]),
    )
    result: mo.RigidRegistrationResult = rigid.register()
    mi.save("data/source.obj", source)
    mi.save("data/target.obj", target)
    mi.save("data/rigid.obj", mo.transform(source, result.transformation))


mu.cli.auto_run()(main)
