import functools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pydantic
import pyvista as pv
import torch
import trimesh.transformations as tf
from loguru import logger

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


@mkit.logging.log_time
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
    logger.remove()
    logger.add(
        sys.stderr,
        level=cfg.log_level,
        filter={
            "mkit.ops.registration.non_rigid.amberg_pytorch3d": "INFO",
            "trimesh.constants": "INFO",
            **mkit.logging.DEFAULT_FILTER,
        },
    )
    torch.set_default_device("cuda")
    template_face: pv.PolyData = prepare_template("face")
    template_mandible: pv.PolyData = prepare_template("mandible")
    template_maxilla: pv.PolyData = prepare_template("maxilla")
    dataset: mkit.io.DICOMDataset = mkit.io.DICOMDataset(cfg.data_dir)
    for patient in dataset.values():
        if len(patient) < 2:
            logger.warning("Patient {} has less than 2 acquisitions", patient.id)
            continue
        for acq_idx, acq in enumerate(patient):
            ic(patient.id, acq.date)
            if (
                Path(
                    f"data/patients/{patient.id}/{acq.date}/02-face-non-rigid.vtp"
                ).exists()
                and Path(
                    f"data/patients/{patient.id}/{acq.date}/02-mandible-non-rigid.vtp"
                ).exists()
                and Path(
                    f"data/patients/{patient.id}/{acq.date}/02-maxilla-non-rigid.vtp"
                ).exists()
            ):
                logger.info("Output exists: {}, {}", patient.id, acq.date)
                continue
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
            skull: pv.PolyData = ct_to_surface(acq.ct, 200)
            face: pv.PolyData = ct_to_surface(acq.ct, -200)
            skull_rigid_result: mkit.ops.registration.RigidRegistrationResult = (
                rigid_registration(template_maxilla, skull)
            )
            rigid_maxilla: pv.PolyData = template_maxilla.transform(
                skull_rigid_result.transform, inplace=False
            )
            non_rigid_maxilla: pv.PolyData = non_rigid_registration(
                rigid_maxilla, skull, "maxilla"
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
            non_rigid_mandible: pv.PolyData = non_rigid_registration(
                rigid_mandible, skull, "mandible"
            )
            face_rigid_result: mkit.ops.registration.RigidRegistrationResult = (
                rigid_registration(template_face, face)
            )
            face_rigid: pv.PolyData = template_face.transform(
                face_rigid_result.transform, inplace=False
            )

            face_non_rigid: pv.PolyData = non_rigid_registration(
                face_rigid, face, "face"
            )

            if acq_idx > 0:
                to_ref: mkit.ops.registration.RigidRegistrationResult = (
                    rigid_registration(non_rigid_maxilla, template_maxilla)
                )
                skull.transform(to_ref.transform, inplace=True)
                rigid_maxilla.transform(to_ref.transform, inplace=True)
                non_rigid_maxilla.transform(to_ref.transform, inplace=True)
                rigid_mandible.transform(to_ref.transform, inplace=True)
                non_rigid_mandible.transform(to_ref.transform, inplace=True)
                face.transform(to_ref.transform, inplace=True)
                face_rigid.transform(to_ref.transform, inplace=True)
                face_non_rigid.transform(to_ref.transform, inplace=True)

            mkit.io.save(skull, f"data/patients/{patient.id}/{acq.date}/00-skull.vtp")
            mkit.io.save(face, f"data/patients/{patient.id}/{acq.date}/00-face.vtp")
            mkit.io.save(
                rigid_maxilla,
                f"data/patients/{patient.id}/{acq.date}/01-maxilla-rigid.vtp",
            )
            mkit.io.save(
                non_rigid_maxilla,
                f"data/patients/{patient.id}/{acq.date}/02-maxilla-non-rigid.vtp",
            )
            mkit.io.save(
                rigid_mandible,
                f"data/patients/{patient.id}/{acq.date}/01-mandible-rigid.vtp",
            )
            mkit.io.save(
                non_rigid_mandible,
                f"data/patients/{patient.id}/{acq.date}/02-mandible-non-rigid.vtp",
            )
            mkit.io.save(
                face_rigid, f"data/patients/{patient.id}/{acq.date}/01-face-rigid.vtp"
            )
            mkit.io.save(
                face_non_rigid,
                f"data/patients/{patient.id}/{acq.date}/02-face-non-rigid.vtp",
            )


mkit.cli.auto_run(log_level="DEBUG")(main)
