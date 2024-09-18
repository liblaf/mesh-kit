from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh as tm
import trimesh.transformations as tt

import mkit
import mkit.ops.registration as reg
import mkit.typing.numpy as nt


class Config(mkit.cli.BaseConfig):
    dataset: Path


def main(cfg: Config) -> None:
    dataset: mkit.io.DICOMDataset = mkit.io.DICOMDataset(cfg.dataset)
    template_face: pv.PolyData = mkit.ext.sculptor.get_template_face()
    template_skull: pv.PolyData = mkit.ext.sculptor.get_template_skull()
    template_maxilla: pv.PolyData = mkit.ext.sculptor.get_template_maxilla()
    template_mandible: pv.PolyData = mkit.ext.sculptor.get_template_mandible()
    for patient in dataset.values():
        for acq in patient:
            ic(acq.meta)
            dpath: Path = Path("data") / patient.meta.id / acq.meta.date
            face: pv.PolyData = pv.read(dpath / "00-face.ply")
            skull: pv.PolyData = pv.read(dpath / "00-skull.ply")
            template_skull.save(dpath / "template.ply")
            result: reg.RigidRegistrationResult = reg.rigid_registration(
                skull,
                template_skull,
                init_global=tt.rotation_matrix(np.pi, [1, 0, 0]),
                inverse=True,
            )
            result = reg.rigid_registration(
                skull,
                template_maxilla,
                init=result.transform,
                inverse=True,
            )
            transform: nt.D44 = remove_scale(result.transform)
            skull.transform(transform, progress_bar=True)
            result = reg.rigid_registration(template_face)
            skull.save(dpath / "01-skull.ply")


def remove_scale(transform: nt.D44Like) -> nt.D44:
    shear: nt.D3Like
    angles: nt.D3Like
    translate: nt.D3Like
    perspective: nt.D4Like
    _scale, shear, angles, translate, perspective = tm.transformations.decompose_matrix(
        transform
    )
    ic(_scale)
    return tm.transformations.compose_matrix(
        shear=shear, angles=angles, translate=translate, perspective=perspective
    )


mkit.cli.auto_run()(main)
