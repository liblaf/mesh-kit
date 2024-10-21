import pydantic

import mkit.struct as ms


# use PascalCase for consistency with DICOM
class DICOMMeta(pydantic.BaseModel):
    AcquisitionDate: ms.dicom.Date
    PatientAge: int
    PatientBirthDate: ms.dicom.Date
    PatientID: str
    PatientName: str
    PatientSex: str
