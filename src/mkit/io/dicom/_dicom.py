import datetime
import functools
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pydicom
import pyvista as pv

import mkit
from mkit.io.dicom._meta import AcquisitionMeta, DatasetMeta, PatientMeta
from mkit.typing import StrPath


class Acquisition:
    dpath: Path

    def __init__(self, dpath: StrPath) -> None:
        self.dpath = Path(dpath)

    @property
    def ct(self) -> pv.ImageData:
        ct: pv.ImageData = pv.read(self.dpath, force_ext=".dcm")
        return ct

    @property
    def date(self) -> datetime.date:
        return datetime.datetime.strptime(self.meta.date, "%Y%m%d").astimezone().date()

    @property
    def dirfile(self) -> pydicom.FileDataset:
        return pydicom.dcmread(self.dirfile_fpath)

    @property
    def dirfile_fpath(self) -> Path:
        return self.dpath / "DIRFILE"

    @property
    def id(self) -> str:
        return self.meta.id

    @functools.cached_property
    def meta(self) -> AcquisitionMeta:
        data: pydicom.FileDataset = pydicom.dcmread(
            self.dpath
            / self.dirfile["DirectoryRecordSequence"][0]["ReferencedFileID"][-1]
        )
        return AcquisitionMeta(
            age=data["PatientAge"].value,
            birth_date=data["PatientBirthDate"].value,
            date=data["AcquisitionDate"].value,
            id=data["PatientID"].value,
            name=str(data["PatientName"].value),
            sex=data["PatientSex"].value,
            time=data["AcquisitionTime"].value,
        )


class Patient(Sequence[Acquisition]):
    dpath: Path

    def __init__(self, dpath: StrPath) -> None:
        self.dpath = Path(dpath)

    def __getitem__(self, idx: int) -> Acquisition:  # pyright: ignore [reportIncompatibleMethodOverride]
        acq_meta: AcquisitionMeta = self.meta.acquisitions[idx]
        return Acquisition(self.dpath / acq_meta.date)

    def __len__(self) -> int:
        return len(self.meta.acquisitions)

    @property
    def id(self) -> str:
        return self.meta.id

    @functools.cached_property
    def meta(self) -> PatientMeta:
        return mkit.utils.load_pydantic(PatientMeta, self.dpath / "patient.json")


class DICOMDataset(Mapping[str, Patient]):
    dpath: Path

    def __init__(self, dpath: StrPath) -> None:
        self.dpath = Path(dpath)

    def __getitem__(self, patient_id: str) -> Patient:
        return Patient(self.dpath / patient_id)

    def __iter__(self) -> Iterator[str]:
        yield from sorted(self.meta.patients.keys())

    def __len__(self) -> int:
        return len(self.meta.patients)

    @functools.cached_property
    def meta(self) -> DatasetMeta:
        return mkit.utils.load_pydantic(DatasetMeta, self.dpath / "dataset.json")
