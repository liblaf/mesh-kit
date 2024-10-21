import pydantic

import mkit.struct as ms


# use PascalCase for consistency with DICOM
class PatientMeta(pydantic.BaseModel):
    Acquisitions: list[ms.DICOMMeta] = []
    PatientBirthDate: ms.dicom.Date
    PatientID: str
    PatientName: str
    PatientSex: str


class DatasetMeta(pydantic.BaseModel):
    Patients: dict[str, PatientMeta] = {}
