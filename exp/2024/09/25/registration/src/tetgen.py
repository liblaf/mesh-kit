from pathlib import Path

import numpy as np
import pydantic
import pyvista as pv

import mkit


class Config(mkit.cli.BaseConfig):
    data_dir: pydantic.DirectoryPath = Path("data/patients/")
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
            continue
        pre_face.point_data["pin_disp"] = np.full(
            (pre_face.n_points, 3), np.nan, np.float32
        )
        pre_face.point_data["pin_mask"] = np.zeros((pre_face.n_points,), dtype=bool)
        pre_maxilla.point_data["pin_disp"] = post_maxilla.points - pre_maxilla.points
        pre_maxilla.point_data["pin_mask"] = np.ones(
            (pre_maxilla.n_points,), dtype=bool
        )
        pre_mandible.point_data["pin_disp"] = post_mandible.points - pre_mandible.points
        pre_mandible.point_data["pin_mask"] = np.ones(
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
                    "pin_mask": pre.point_data["pin_mask"],
                },
                method=mkit.ops.transfer.P2PAuto(),
            )
        )
        mkit.io.save(tetra, f"{cfg.data_dir}/{patient.id}/pre.vtu")
        pre = tetra.extract_surface()
        mkit.io.save(pre, f"{cfg.data_dir}/{patient.id}/pre.vtp")


mkit.cli.auto_run()(main)
