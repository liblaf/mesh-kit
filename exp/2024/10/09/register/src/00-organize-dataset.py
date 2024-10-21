import shutil
from pathlib import Path

import pydantic

import mkit.io as mi
import mkit.struct as ms
import mkit.utils as mu


class Config(mu.cli.BaseConfig):
    raw: pydantic.DirectoryPath = Path("~/data/01-raw/00-CT资料").expanduser()
    output: Path = Path("~/data/01-raw/01-CT").expanduser()


def main(cfg: Config) -> None:
    dataset_meta: ms.dicom_dataset.DatasetMeta = ms.dicom_dataset.DatasetMeta()
    for fpath in cfg.raw.rglob("DIRFILE"):
        dpath: Path = fpath.parent
        dicom: ms.DICOM = mi.mkit.load_dicom(dpath)
        ic(dicom.patient_id, dicom.acquisition_date)
        output: Path = (
            cfg.output
            / dicom.patient_id
            / ms.dicom_dataset.format_date(dicom.acquisition_date)
        )
        if dicom.patient_id not in dataset_meta.Patients:
            dataset_meta.Patients[dicom.patient_id] = ms.dicom_dataset.PatientMeta(
                **dicom.meta.model_dump()
            )
        dataset_meta.Patients[dicom.patient_id].Acquisitions.append(dicom.meta)
        shutil.copytree(dpath, output, dirs_exist_ok=True)
    for patient_meta in dataset_meta.Patients.values():
        patient_dpath: Path = cfg.output / patient_meta.PatientID
        mu.save_pydantic(patient_dpath / "patient.json", patient_meta)
    mu.save_pydantic(cfg.output / "dataset.json", dataset_meta)


mu.cli.auto_run()(main)
