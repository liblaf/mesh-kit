import collections
import concurrent.futures
import shutil
from pathlib import Path
from typing import Literal

import pydantic
import pydicom

import mkit


class CLIConfig(mkit.cli.BaseConfig):
    raw: Path
    output: Path


class Acquisition(pydantic.BaseModel):
    age: str
    birth_date: str
    date: str
    id: str
    name: str
    sex: Literal["F", "M"]
    time: str


class Patient(pydantic.BaseModel):
    acquisitions: list[Acquisition]
    birth_date: str
    id: str
    name: str
    sex: Literal["F", "M"]


class Dataset(pydantic.BaseModel):
    patients: dict[str, Patient]


def extract_meta(data: pydicom.FileDataset) -> Acquisition:
    return Acquisition(
        age=data["PatientAge"].value,
        birth_date=data["PatientBirthDate"].value,
        date=data["AcquisitionDate"].value,
        id=data["PatientID"].value,
        name=str(data["PatientName"].value),
        sex=data["PatientSex"].value,
        time=data["AcquisitionTime"].value,
    )


def process_acquisition(dirfile_fpath: Path, output_dir: Path) -> Acquisition:
    dirfile: pydicom.FileDataset = pydicom.dcmread(dirfile_fpath)
    seq: pydicom.DataElement = dirfile["DirectoryRecordSequence"]
    file_id: pydicom.DataElement = seq[0]["ReferencedFileID"]
    record_fpath: Path = dirfile_fpath.with_name(file_id[-1])
    record: pydicom.FileDataset = pydicom.dcmread(record_fpath)
    meta: Acquisition = extract_meta(record)
    ic(meta)
    dpath: Path = output_dir / meta.id / meta.date
    dpath.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dirfile_fpath.parent, dpath, dirs_exist_ok=True)
    return meta


@mkit.cli.auto_run()
def main(cfg: CLIConfig) -> None:
    patients: dict[str, list[Acquisition]] = collections.defaultdict(list)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures: list[concurrent.futures.Future[Acquisition]] = [
            executor.submit(process_acquisition, fpath, cfg.output)
            for fpath in cfg.raw.rglob("DIRFILE")
        ]
        for future in concurrent.futures.as_completed(futures):
            meta: Acquisition = future.result()
            patients[meta.id].append(meta)
    dataset = Dataset(patients={})
    for id_, acquisitions in patients.items():
        acquisitions.sort(key=lambda x: x.date)
        fpath: Path = cfg.output / id_ / "patient.json"
        patient: Patient = Patient(
            acquisitions=acquisitions,
            birth_date=acquisitions[0].birth_date,
            id=id_,
            name=acquisitions[0].name,
            sex=acquisitions[0].sex,
        )
        fpath.write_text(patient.model_dump_json())
        dataset.patients[id_] = patient
    (cfg.output / "dataset.json").write_text(dataset.model_dump_json())
