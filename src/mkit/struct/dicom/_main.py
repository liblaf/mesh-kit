from __future__ import annotations

import datetime
import functools
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pydicom
import pydicom.valuerep

import mkit.io as mi
import mkit.struct as ms
import mkit.typing as mt

if TYPE_CHECKING:
    import pyvista as pv


class DICOM:
    _path: Path

    def __init__(self, path: mt.StrPath) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    @functools.cached_property
    def image_data(self) -> pv.ImageData:
        return mi.pyvista.load_image_data(self.path)

    def extract_contour(
        self,
        threshold: float,
        radius_factor: float = 1.5,
        std_dev: float = 2.0,
        connectivity: Literal[
            "all",
            "largest",
            "specified",
            "cell_seed",
            "point_seed",
            "closest",
        ]
        | None = "largest",
    ) -> pv.PolyData:
        data: pv.ImageData = self.image_data.gaussian_smooth(
            radius_factor=radius_factor, std_dev=std_dev
        )  # # pyright: ignore [reportAssignmentType]
        contour: pv.PolyData = data.contour([threshold], compute_scalars=False)  # pyright: ignore [reportArgumentType, reportAssignmentType]
        if connectivity:
            contour.connectivity("largest", inplace=True)
            del contour.point_data["RegionId"]
            del contour.cell_data["RegionId"]
        return contour

    def extract_face(
        self,
        threshold: float = -200,
        radius_factor: float = 1.5,
        std_dev: float = 2.0,
        connectivity: Literal[
            "all",
            "largest",
            "specified",
            "cell_seed",
            "point_seed",
            "closest",
        ]
        | None = "largest",
    ) -> pv.PolyData:
        return self.extract_contour(
            threshold=threshold,
            radius_factor=radius_factor,
            std_dev=std_dev,
            connectivity=connectivity,
        )

    def extract_skull(
        self,
        threshold: float = 200,
        radius_factor: float = 1.5,
        std_dev: float = 2.0,
        connectivity: Literal[
            "all",
            "largest",
            "specified",
            "cell_seed",
            "point_seed",
            "closest",
        ]
        | None = "largest",
    ) -> pv.PolyData:
        return self.extract_contour(
            threshold=threshold,
            radius_factor=radius_factor,
            std_dev=std_dev,
            connectivity=connectivity,
        )

    @functools.cached_property
    def dirfile(self) -> pydicom.FileDataset:
        return pydicom.dcmread(self.path / "DIRFILE")

    @functools.cached_property
    def first_file_dataset(self) -> pydicom.FileDataset:
        filename: str = self.dirfile["DirectoryRecordSequence"].value[0][
            "ReferencedFileID"
        ][-1]
        return pydicom.dcmread(self.path / filename)

    @functools.cached_property
    def acquisition_date(self) -> datetime.date:
        return datetime.datetime.strptime(
            self.first_file_dataset["AcquisitionDate"].value, "%Y%m%d"
        ).date()

    @functools.cached_property
    def patient_name(self) -> str:
        name: pydicom.valuerep.PersonName = self.first_file_dataset["PatientName"].value
        return str(name)

    @functools.cached_property
    def patient_id(self) -> str:
        return self.first_file_dataset["PatientID"].value

    @functools.cached_property
    def patient_birth_date(self) -> datetime.date:
        return datetime.datetime.strptime(
            self.first_file_dataset["PatientBirthDate"].value, "%Y%m%d"
        ).date()

    @functools.cached_property
    def patient_sex(self) -> Literal["F", "M"]:
        return self.first_file_dataset["PatientSex"].value

    @functools.cached_property
    def patient_age(self) -> int:
        age_str: str = self.first_file_dataset["PatientAge"].value
        return int(age_str.removesuffix("Y"))

    @functools.cached_property
    def meta(self) -> ms.DICOMMeta:
        return ms.DICOMMeta(
            AcquisitionDate=self.acquisition_date,
            PatientAge=self.patient_age,
            PatientBirthDate=self.patient_birth_date,
            PatientID=self.patient_id,
            PatientName=self.patient_name,
            PatientSex=self.patient_sex,
        )
