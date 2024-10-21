import datetime
import functools
from collections.abc import Sequence
from pathlib import Path
from typing import overload

import mkit.io as mi
import mkit.struct as ms
import mkit.typing as mt


class DICOMPatient(Sequence[ms.DICOM]):
    _path: Path

    def __init__(self, path: mt.StrPath) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def __len__(self) -> int:
        return len(self.meta.Acquisitions)

    @overload
    def __getitem__(self, idx: int) -> ms.DICOM: ...
    @overload
    def __getitem__(self, idx: slice) -> list[ms.DICOM]: ...
    def __getitem__(self, idx: int | slice) -> ms.DICOM | list[ms.DICOM]:
        if isinstance(idx, slice):
            return [
                self._get_acquisition(a.AcquisitionDate)
                for a in self.meta.Acquisitions[idx]
            ]
        return self._get_acquisition(self.meta.Acquisitions[idx].AcquisitionDate)

    def _get_acquisition(self, date: datetime.date) -> ms.DICOM:
        return mi.mkit.load_dicom(self.path / ms.dicom_dataset.format_date(date))

    @property
    def patient_id(self) -> str:
        return self.meta.PatientID

    @property
    def patient_sex(self) -> str:
        return self.meta.PatientSex

    @property
    def patient_birth_date(self) -> datetime.date:
        return self.meta.PatientBirthDate

    @property
    def patient_name(self) -> str:
        return self.meta.PatientName

    @functools.cached_property
    def meta(self) -> ms.dicom_dataset.PatientMeta:
        meta_fpath: Path = self.path / "patient.json"
        return ms.dicom_dataset.PatientMeta.model_validate_json(meta_fpath.read_text())
