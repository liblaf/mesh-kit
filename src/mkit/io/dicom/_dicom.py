import functools
from collections.abc import Mapping, Sequence
from pathlib import Path

from mkit.io.dicom._meta import MetaDataset
from mkit.typing import StrPath


class Acquisition:
    dpath: Path

    def __init__(self, dpath: StrPath) -> None:
        self.dpath = Path(dpath)


class Patient(Sequence[Acquisition]):
    dpath: Path

    def __init__(self, dpath: StrPath) -> None:
        self.dpath = Path(dpath)


class DICOMDataset(Mapping[str, Patient]):
    dpath: Path

    def __init__(self, dpath: StrPath) -> None:
        self.dpath = Path(dpath)

    @functools.cached_property
    def meta(self) -> MetaDataset:
        return MetaDataset.model_validate_json(
            (self.dpath / "dataset.json").read_text()
        )
