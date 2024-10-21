import functools
from collections.abc import Iterator, Mapping
from pathlib import Path

import mkit.struct as ms
import mkit.typing as mt


class DICOMDataset(Mapping[str, ms.DICOMPatient]):
    _path: Path

    def __init__(self, path: mt.StrPath) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def __getitem__(self, patient_id: str) -> ms.DICOMPatient:
        return ms.DICOMPatient(self._path / patient_id)

    def __iter__(self) -> Iterator[str]:
        yield from self.meta.Patients.keys()

    def __len__(self) -> int:
        return len(self.meta.Patients)

    @functools.cached_property
    def meta(self) -> ms.dicom_dataset.DatasetMeta:
        meta_fpath: Path = self.path / "dataset.json"
        return ms.dicom_dataset.DatasetMeta.model_validate_json(meta_fpath.read_text())
