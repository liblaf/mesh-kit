from pathlib import Path

import numpy as np
import pydantic
import pyvista as pv
from loguru import logger

import mkit


class Config(mkit.cli.BaseConfig):
    data_dir: pydantic.DirectoryPath = Path("data/patients/")
    output_dir: Path = Path("data/tetgen/")
    dicom_dir: pydantic.DirectoryPath = Path("~/Documents/CT/").expanduser()


def main(cfg: Config) -> None:
    dataset: mkit.io.DICOMDataset = mkit.io.DICOMDataset(cfg.dicom_dir)
    for patient in dataset.values():
        try:
            pre_face: pv.PolyData = mkit.io.pyvista.load_poly_data(
                f"{cfg.data_dir}/{patient.id}/{patient[0].date}/02-face-non-rigid.vtp"
            )
            pre_mandible: pv.PolyData = mkit.io.pyvista.load_poly_data(
                f"{cfg.data_dir}/{patient.id}/{patient[0].date}/02-mandible-non-rigid.vtp"
            )
            pre_maxilla: pv.PolyData = mkit.io.pyvista.load_poly_data(
                f"{cfg.data_dir}/{patient.id}/{patient[0].date}/02-maxilla-non-rigid.vtp"
            )
            post_face: pv.PolyData = mkit.io.pyvista.load_poly_data(
                f"{cfg.data_dir}/{patient.id}/{patient[-1].date}/02-face-non-rigid.vtp"
            )
            post_mandible: pv.PolyData = mkit.io.pyvista.load_poly_data(
                f"{cfg.data_dir}/{patient.id}/{patient[-1].date}/02-mandible-non-rigid.vtp"
            )
            post_maxilla: pv.PolyData = mkit.io.pyvista.load_poly_data(
                f"{cfg.data_dir}/{patient.id}/{patient[-1].date}/02-maxilla-non-rigid.vtp"
            )
        except FileNotFoundError:
            logger.warning(f"Patient {patient.id} has not been registered.")
            continue
        output_fpath: Path = cfg.output_dir / patient.id / "pre.vtu"
        if output_fpath.exists():
            logger.info("Output exists: {}", output_fpath)
            continue
        pre_face.point_data["pin_disp"] = np.zeros((pre_face.n_points, 3), np.float32)
        pre_face.point_data["pin_mask"] = np.zeros((pre_face.n_points,), dtype=bool)
        pre_face.point_data["face_mask"] = np.ones((pre_face.n_points,), dtype=bool)
        pre_maxilla.point_data["pin_disp"] = post_maxilla.points - pre_maxilla.points
        pre_maxilla.point_data["pin_mask"] = np.ones(
            (pre_maxilla.n_points,), dtype=bool
        )
        pre_maxilla.point_data["face_mask"] = np.zeros(
            (pre_maxilla.n_points,), dtype=bool
        )
        pre_mandible.point_data["pin_disp"] = post_mandible.points - pre_mandible.points
        pre_mandible.point_data["pin_mask"] = np.ones(
            (pre_mandible.n_points,), dtype=bool
        )
        pre_mandible.point_data["face_mask"] = np.zeros(
            (pre_mandible.n_points,), dtype=bool
        )
        pre_skull: pv.PolyData = pv.merge([pre_maxilla, pre_mandible])
        pre_skull.flip_normals()
        pre: pv.PolyData = pv.merge([pre_face, pre_skull])
        tetra: pv.UnstructuredGrid = mkit.ext.tetwild(pre)
        tetra.point_data.update(
            mkit.ops.transfer.surface_to_volume(
                pre,
                tetra,
                {
                    "pin_disp": pre.point_data["pin_disp"],
                    "Weight": pre.point_data["Weight"],
                },
                method=mkit.ops.transfer.P2PAuto(fill_value=0.0),
            )
        )
        tetra.point_data.update(
            mkit.ops.transfer.surface_to_volume(
                pre,
                tetra,
                {
                    "pin_mask": pre.point_data["pin_mask"],
                    "face_mask": pre.point_data["face_mask"],
                },
                method=mkit.ops.transfer.P2PAuto(fill_value=False),
            )
        )
        pre = tetra.extract_surface()
        pre_face = pre.extract_points(pre.point_data["face_mask"]).extract_surface()
        pre_skull = pre.extract_points(~pre.point_data["face_mask"]).extract_surface()
        mkit.io.save(tetra, output_fpath)
        mkit.io.save(pre, output_fpath.with_suffix(".vtp"))
        mkit.io.save(pre_face, output_fpath.with_name("pre-face.vtp"))
        mkit.io.save(pre_skull, output_fpath.with_name("pre-skull.vtp"))


mkit.cli.auto_run()(main)
