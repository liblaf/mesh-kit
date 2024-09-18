import functools
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pydicom
import pyvista as pv

import mkit
from mkit.io.dicom._meta import MetaAcquisition, MetaDataset, MetaPatient
from mkit.typing import StrPath


class Acquisition:
    dpath: Path

    def __init__(self, dpath: StrPath) -> None:
        self.dpath = Path(dpath)

    @property
    def ct(self) -> pv.ImageData:
        ct: pv.ImageData = pv.read(self.dpath, force_ext=".dcm")
        return ct

    @functools.cached_property
    def meta(self) -> MetaAcquisition:
        data: pydicom.FileDataset = pydicom.dcmread(self.dpath / "I10")
        return MetaAcquisition(
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
        meta_acq: MetaAcquisition = self.meta.acquisitions[idx]
        return Acquisition(self.dpath / meta_acq.date)

    def __len__(self) -> int:
        return len(self.meta.acquisitions)

    @functools.cached_property
    def meta(self) -> MetaPatient:
        return mkit.utils.load_pydantic(MetaPatient, self.dpath / "patient.json")


class DICOMDataset(Mapping[str, Patient]):
    dpath: Path

    def __init__(self, dpath: StrPath) -> None:
        self.dpath = Path(dpath)

    def __getitem__(self, patient_id: str) -> Patient:
        return Patient(self.dpath / patient_id)

    def __iter__(self) -> Iterator[str]:
        yield from self.meta.patients.keys()

    def __len__(self) -> int:
        return len(self.meta.patients)

    @functools.cached_property
    def meta(self) -> MetaDataset:
        return mkit.utils.load_pydantic(MetaDataset, self.dpath / "dataset.json")
