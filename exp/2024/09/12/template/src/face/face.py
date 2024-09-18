from pathlib import Path
from typing import TYPE_CHECKING

import mkit

if TYPE_CHECKING:
    import pyvista as pv

    from mkit.ops.registration import RigidRegistrationResult


class Config(mkit.cli.BaseConfig):
    template: Path = Path(
        "~/.local/opt/Wrap/Gallery/TexturingXYZ/XYZ_ReadyToSculpt_eyesOpen_PolyGroups_GEO.obj"
    ).expanduser()


@mkit.cli.auto_run()
def main(cfg: Config) -> None:
    source: pv.PolyData = mkit.io.pyvista.read_poly_data(cfg.template)
    ic(source.cell_data["GroupIds"])
    ic(source.field_data["GroupNames"])
    group_id_to_delete: list[int] = [
        i
        for i in range(len(source.field_data["GroupNames"]))
        if source.field_data["GroupNames"][i]
        in [
            "MouthSocketBottom",
            "MouthSocketTop",
            "EyeSocketTop",
            "EyeSocketBottom",
        ]
    ]
    unstructured: pv.UnstructuredGrid = source.extract_values(
        group_id_to_delete,
        scalars="GroupIds",
        preference="cell",
        invert=True,
        progress_bar=True,
    )
    ic(unstructured)
    source = unstructured.extract_surface(progress_bar=True)
    target: pv.PolyData = mkit.ext.sculptor.get_template_face()
    target.save("data/target.obj")
    result: RigidRegistrationResult = mkit.ops.registration.rigid_registration(
        source, target
    )
    source.transform(result.transform, inplace=True, progress_bar=True)
    source.save("data/aligned.obj")
