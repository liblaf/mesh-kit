from pathlib import Path
from typing import TYPE_CHECKING

import mkit
import mkit.typing.numpy as nt
import numpy as np
import pyvista as pv
from mkit.typing import StrPath

if TYPE_CHECKING:
    import open3d as o3d
    from mkit.ops.registration._global._result import GlobalRegistrationResult


class Config(mkit.cli.BaseConfig):
    template: Path


def read_obj(fpath: StrPath) -> pv.PolyData:
    fpath: Path = Path(fpath)
    group_dup: list[str] = []
    for line in mkit.utils.strip_comments(fpath.read_text()):
        if line.startswith("g"):
            name: str
            _, name = line.split()
            group_dup.append(name)
    group_uniq: list[str] = list(set(group_dup))
    dup_to_uniq: dict[str, int] = {name: i for i, name in enumerate(group_uniq)}
    mesh: pv.PolyData = pv.read(fpath)
    group_id: nt.IN = mkit.math.numpy.cast(mesh.cell_data["GroupIds"], int)
    group_id = np.asarray([dup_to_uniq[group_dup[i]] for i in group_id])
    mesh.cell_data["GroupIds"] = group_id
    mesh.field_data["GroupNames"] = group_uniq
    return mesh


@mkit.cli.auto_run()
def main(cfg: Config) -> None:
    if False:
        demo = o3d.data.DemoICPPointClouds()
        source: o3d.geometry.PointCloud = o3d.io.read_point_cloud(demo.paths[0])
        source_pv: pv.PolyData = pv.wrap(np.asarray(source.points))
        target: o3d.geometry.PointCloud = o3d.io.read_point_cloud(demo.paths[1])
        target_pv: pv.PolyData = pv.wrap(np.asarray(target.points))
        ic(0.05 / source_pv.length)
        ic(0.05 / target_pv.length)
        return
    source: pv.PolyData = read_obj(cfg.template)
    target: pv.PolyData = mkit.ext.sculptor.get_template_skull()
    source.save("data/source.vtp")
    target.save("data/target.vtp")
    result: GlobalRegistrationResult = mkit.ops.registration.global_registration(
        source, target
    )
    source.transform(result.transform, inplace=True)
    ic(np.unique(source.cell_data["GroupIds"]))
    ic(source.field_data["GroupNames"])
    source.save("data/result.vtp")
