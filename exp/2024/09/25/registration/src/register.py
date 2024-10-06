import functools
from pathlib import Path
from typing import Any

import numpy as np
import pydantic
import pyvista as pv

import mkit
import mkit.typing.numpy as nt


class Config(mkit.cli.BaseConfig):
    data_dir: pydantic.DirectoryPath = Path("/home/liblaf/Documents/CT")


def ct_to_surface(ct: pv.ImageData, threshold: float) -> pv.PolyData:
    ct = ct.gaussian_smooth()
    contours: pv.PolyData = ct.contour([threshold])
    contour: pv.PolyData = contours.connectivity("largest")
    return contour


@functools.cache
def prepare_template(component: str) -> pv.PolyData:
    source: pv.PolyData = mkit.io.pyvista.load_poly_data(
        f"data/template/{component}.vtp"
    )
    return source


@functools.cache
def load_non_rigid_params(component: str) -> Any:
    cfg: Any = mkit.utils.load_yaml(f"params/{component}.yaml")
    return cfg["non-rigid"]


def rigid_registration(
    source: pv.PolyData, target: pv.PolyData, *, estimate_init: bool = True
) -> mkit.ops.registration.RigidRegistrationResult:
    result: mkit.ops.registration.rigid.RigidRegistrationResult = (
        mkit.ops.registration.rigid.rigid_registration(
            source,
            target,
            estimate_init=estimate_init,
            source_weight=source.point_data["Weight"],
        )
    )
    return result


def non_rigid_registration(
    source: pv.PolyData, target: pv.PolyData, component: str
) -> pv.PolyData:
    weight: nt.FN = np.array(source.point_data["Weight"])
    weight[source.points[:, 2] > target.bounds[5]] = 0
    non_rigid: mkit.ops.registration.NonRigidRegistrationResult = (
        mkit.ops.registration.non_rigid.non_rigid_registration(
            mkit.ops.registration.non_rigid.Amberg(
                source,
                target,
                point_data={"weight": weight},
                steps=load_non_rigid_params(component)["steps"],
            )
        )
    )
    result: pv.PolyData = source.copy()
    result.points = non_rigid.points
    return result


def main(cfg: Config) -> None:
    template_face: pv.PolyData = prepare_template("face")
    template_mandible: pv.PolyData = prepare_template("mandible")
    template_maxilla: pv.PolyData = prepare_template("maxilla")
    dataset: mkit.io.DICOMDataset = mkit.io.DICOMDataset(cfg.data_dir)
    for patient in dataset.values():
        for acq_idx, acq in enumerate(patient):
            if acq_idx > 0:
                template_face = mkit.io.pyvista.load_poly_data(
                    f"data/patients/{patient.id}/{patient[0].date}/02-face-non-rigid.vtp"
                )
                template_mandible = mkit.io.pyvista.load_poly_data(
                    f"data/patients/{patient.id}/{patient[0].date}/02-mandible-non-rigid.vtp"
                )
                template_maxilla = mkit.io.pyvista.load_poly_data(
                    f"data/patients/{patient.id}/{patient[0].date}/02-maxilla-non-rigid.vtp"
                )
            else:
                template_face = prepare_template("face")
                template_mandible = prepare_template("mandible")
                template_maxilla = prepare_template("maxilla")
            face: pv.PolyData = ct_to_surface(acq.ct, -200)
            mkit.io.save(face, f"data/patients/{patient.id}/{acq.date}/00-face.vtp")
            face_rigid_result: mkit.ops.registration.RigidRegistrationResult = (
                rigid_registration(template_face, face)
            )
            face_rigid: pv.PolyData = template_face.transform(
                face_rigid_result.transform, inplace=False
            )
            mkit.io.save(
                face_rigid, f"data/patients/{patient.id}/{acq.date}/01-face-rigid.vtp"
            )
            face_non_rigid: pv.PolyData = non_rigid_registration(
                face_rigid, face, "face"
            )
            mkit.io.save(
                face_non_rigid,
                f"data/patients/{patient.id}/{acq.date}/02-face-non-rigid.vtp",
            )

            skull: pv.PolyData = ct_to_surface(acq.ct, 200)
            mkit.io.save(skull, f"data/patients/{patient.id}/{acq.date}/00-skull.vtp")
            skull_rigid_result: mkit.ops.registration.RigidRegistrationResult = (
                rigid_registration(template_maxilla, skull)
            )
            rigid_maxilla: pv.PolyData = template_maxilla.transform(
                skull_rigid_result.transform, inplace=False
            )
            mkit.io.save(
                rigid_maxilla,
                f"data/patients/{patient.id}/{acq.date}/01-maxilla-rigid.vtp",
            )
            non_rigid_maxilla: pv.PolyData = non_rigid_registration(
                rigid_maxilla, skull, "maxilla"
            )
            mkit.io.save(
                non_rigid_maxilla,
                f"data/patients/{patient.id}/{acq.date}/02-maxilla-non-rigid.vtp",
            )

            rigid_mandible: pv.PolyData = template_mandible.transform(
                skull_rigid_result.transform, inplace=False
            )
            mandible_rigid_result: mkit.ops.registration.RigidRegistrationResult = (
                rigid_registration(rigid_mandible, skull, estimate_init=False)
            )
            rigid_mandible = rigid_mandible.transform(
                mandible_rigid_result.transform, inplace=False
            )
            mkit.io.save(
                rigid_mandible,
                f"data/patients/{patient.id}/{acq.date}/01-mandible-rigid.vtp",
            )
            non_rigid_mandible: pv.PolyData = non_rigid_registration(
                rigid_mandible, skull, "mandible"
            )
            mkit.io.save(
                non_rigid_mandible,
                f"data/patients/{patient.id}/{acq.date}/02-mandible-non-rigid.vtp",
            )


mkit.cli.auto_run()(main)
